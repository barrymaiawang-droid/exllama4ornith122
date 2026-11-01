#!/usr/bin/env python3
"""Portable derivative of ornith122-serve.py (isolated serving protocol, CPU-testable).

v1 keeps the production protocol surface of the original ornith122-serve.py
(external production file, unchanged, not part of this repository; endpoints,
OpenAI-compatible chat/completions shapes, tool-call parsing, SSE
framing) and fixes the gaps demonstrated red -> green by the focused CPU suite
test_ornith_serve.py in this directory:

* per-request budget validation: max_tokens/temperature/top_p and messages are
  checked at the edge (400 invalid_request_error instead of a dropped
  connection or a 500); the effective budget is resolved against the ctx room
  BEFORE any SSE headers are sent (missing -> DEFAULT_MAX_TOKENS clamped to the
  room left in ctx, explicit overflow -> 400 with OpenAI-style token counts),
  and the resolved value is what flows to Job.max_new_tokens;
* generator cancellation: closing the Engine.stream generator (client
  disconnect, error stage, handler cleanup) calls Generator.cancel(job) and
  releases the engine lock immediately instead of leaking the job on the
  backend until garbage collection;
* sampler construction matches the pinned exllamav3 signature
  TopPSampler(top_p=, temperature=) (the production file passes p=, which is a
  TypeError for every temperature > 0 request);
* the content phase of Splitter holds back a marker-sized tail, so a
  <tool_call> marker split across stream deltas can no longer leak its prefix
  (e.g. "<tool_") into visible content;
* a mid-stream engine failure terminates the event stream instead of splicing
  an HTTP/1.1 500 status line into the SSE body.

Portability changes carried over from v0: no global torch import (runtime info
for /props comes from the bound backend), no host-private default model path,
the default listen port is 0 (OS-assigned; never a production port range), and
the exllamav3 binding is lazy and injectable so tests drive the protocol with
explicit fakes (FakeTokenizer / FakeGenerator / FakeJob) instead of a real
engine import.

v2 adds one opt-in speculative mode: `--draft-model <dir>` selects the native
DFlash2 drafter (loaded from its own directory via Config.from_directory →
Model.from_config → Cache before load, depth taken from the drafter's own
caps, default 15) and is mutually exclusive with `--draft`; AR (`--draft 0`)
and native MTP2 (`--draft 2`, still the default) are unchanged. The mode is
labeled "dflash2" in the ready log, /health and /props.

v3 adds one opt-in, per-request feature: `repetition_guard` in the request body turns on
the exllamav3 generator's repetition guard for that request alone - `true` for its
defaults, or an object of options (mode "active"/"shadow", min_match, confirm,
max_interventions, escalate_within, nudge). Absent or false leaves the request exactly as
v2, including the Job kwargs, so an unopted request is byte-identical. Malformed options
are refused with 400 invalid_request_error before any SSE header is sent (the guard itself
validates with bare asserts, which would otherwise fail mid-stream). The guard summary the
engine reports on the final result is returned as `repetition_guard` on the non-stream
response body and on the final stream chunk; completion_tokens stays the visible count,
because this server derives it by re-encoding the visible text and the guard's hidden
nudge tokens never reach it.

Requires a source tree whose Job accepts repetition_guard (exllama4ornith122 b13ec7bc or
later); the kwarg is passed only when the client opts in, so nothing else depends on it.

Endpoints: /health, /props, /v1/models, /v1/chat/completions, /v1/completions.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MAX_TOKENS = 131072
# Output ceiling and omitted-request default. Engine.resolve_budget clamps an
# omitted value to the context room, but rejects an explicit over-ceiling value.

# The repetition guard's option names, mirroring exllamav3/generator/repetition_guard.py
# DEFAULTS. Only these may appear in a request's repetition_guard object.
GUARD_OPTIONS = ("mode", "min_match", "confirm", "max_interventions", "escalate_within",
                 "nudge")

# Sentinel for "the request never mentioned repetition_guard": the fleet default
# (REPETITION_GUARD) applies. An explicit false/null means off for that request and
# must NOT fall back to the default.
_ABSENT = object()

TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
FUNCTION_RE = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)

OPEN_TAGS = ("<think>",)
CLOSE_TAGS = ("</think>", "<｜end▁of▁thinking｜>")
TOOL_OPEN = "<tool_call>"


class RequestError(ValueError):
    """Malformed or invalid client parameter -> HTTP 400 invalid_request_error."""


class BudgetError(ValueError):
    """Request budget does not fit the context window -> HTTP 400 with counts."""


def _int_param(body, name, minimum):
    """Optional integer request parameter; strict type check (bool rejected)."""
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"{name} must be an integer")
    if value < minimum:
        raise RequestError(f"{name} must be an integer >= {minimum}")
    return value


def _float_param(body, name, lo, hi, default, lo_exclusive=False):
    """Optional numeric request parameter with a range check (bool rejected)."""
    value = body.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{name} must be a number")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise RequestError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise RequestError(f"{name} must be a finite number")
    below = number <= lo if lo_exclusive else number < lo
    if below or number > hi:
        rel = ">" if lo_exclusive else ">="
        raise RequestError(f"{name} must be {rel} {lo} and <= {hi}")
    return number


def _guard_param(body):
    """Per-request repetition guard: absent, off, on, or validated options.

    Returns _ABSENT when the body does not carry the field at all, so the caller can
    apply the fleet default; None for an explicit false/null (off for this request,
    even when a default is configured); True for the engine's own defaults; or a dict
    of validated options. The limits mirror exllamav3's own asserts
    (SuffixLoopDetector: min_match >= 2, confirm >= 1; RepetitionGuard: mode in
    {active, shadow}, max_interventions and escalate_within >= 0, no unknown option),
    so anything accepted here cannot trip an assert inside the engine after the
    response has already started.
    """
    if "repetition_guard" not in body:
        return _ABSENT
    value = body["repetition_guard"]
    if value is None or value is False:
        return None
    if value is True:
        return True
    if not isinstance(value, dict):
        raise RequestError("repetition_guard must be true, false or an object")
    unknown = sorted(set(value) - set(GUARD_OPTIONS))
    if unknown:
        raise RequestError("repetition_guard: unknown option(s) " + ", ".join(unknown))
    out = {}
    mode = value.get("mode")
    if mode is not None:
        if mode not in ("active", "shadow"):
            raise RequestError('repetition_guard.mode must be "active" or "shadow"')
        out["mode"] = mode
    for name, minimum in (("min_match", 2), ("confirm", 1),
                          ("max_interventions", 0), ("escalate_within", 0)):
        v = value.get(name)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int):
            raise RequestError(f"repetition_guard.{name} must be an integer")
        if v < minimum:
            raise RequestError(f"repetition_guard.{name} must be an integer >= {minimum}")
        out[name] = v
    if "nudge" in value:
        nudge = value["nudge"]
        if nudge is not None and not isinstance(nudge, str):
            raise RequestError("repetition_guard.nudge must be a string or null")
        out["nudge"] = nudge
    return out


def _guard_default():
    """Fleet-wide repetition guard default from REPETITION_GUARD.

    Unset/empty/0/false/off -> None: off unless a request opts in (the v2 behaviour).
    1/true/on -> the engine defaults. "shadow" -> detect and log only, which is the
    rollout step the engine docs ask for first. Anything else is parsed as JSON and
    validated exactly like the request field, so an unusable value fails at startup
    instead of on the first request.
    """
    raw = os.environ.get("REPETITION_GUARD")
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    low = text.lower()
    if low in ("0", "false", "off", "no"):
        return None
    if low in ("1", "true", "on", "yes"):
        return True
    if low == "shadow":
        return {"mode": "shadow"}
    try:
        parsed = json.loads(text)
    except Exception:
        raise SystemExit(f"REPETITION_GUARD must be 0/1/shadow or JSON: {raw!r}")
    try:
        return _guard_param({"repetition_guard": parsed})
    except RequestError as exc:
        raise SystemExit(f"REPETITION_GUARD invalid: {exc}")


def _report_guard(guard_summary, explicit):
    """Drop a fleet-default summary that detected nothing.

    A client that asked for the guard gets its summary either way, including an
    all-zero one. A client that never mentioned it only gets one when the guard
    actually did something, so a clean response keeps the v2 body shape exactly
    (same keys) even though the deployment is guarded by default.
    """
    if guard_summary is None or explicit:
        return guard_summary
    return guard_summary if guard_summary.get("detections") else None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


SPECIAL_SCRUB = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def scrub(text: str):
    for marker in SPECIAL_SCRUB:
        text = text.replace(marker, "")
    return text


def _xml_payload(value: str):
    """Remove the native template's one framing newline on either side."""
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


def coerce(value: str, expected_type: str | None = None):
    if expected_type == "string":
        # Native XML parameters contain raw text. JSON-looking file contents
        # are still text; decoding them a second time changes the payload.
        return _xml_payload(value)
    v = value.strip()
    try:
        return json.loads(v)
    except Exception:
        return v


def _tool_param_types(tools) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t) if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        props = (fn.get("parameters") or {}).get("properties") or {}
        if isinstance(name, str) and isinstance(props, dict):
            mapping[name] = {
                k: v.get("type")
                for k, v in props.items()
                if isinstance(v, dict) and isinstance(v.get("type"), str)
            }
    return mapping


def parse_tool_calls(text: str, tools=None):
    """Split reply text into (visible_text, [openai tool_call dicts])."""
    if TOOL_OPEN not in text:
        return scrub(text), []
    first = text.find(TOOL_OPEN)
    content = scrub(text[:first].rstrip())
    schema_types = _tool_param_types(tools)
    calls = []
    for block in TOOL_BLOCK_RE.findall(text[first:]):
        for name, body in FUNCTION_RE.findall(block):
            param_types = schema_types.get(name, {})
            args = {p: coerce(v, param_types.get(p)) for p, v in PARAM_RE.findall(body)}
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": name,
                             "arguments": json.dumps(args, ensure_ascii=False)},
            })
    if not calls:
        log("tool markup present but unparsed:", repr(text[first:first + 120]))
    return content, calls


def _partial_suffix(buf: str, markers) -> int:
    """Length of the longest tail of buf that is a proper prefix of one of markers."""
    best = 0
    for m in markers:
        for k in range(min(len(m) - 1, len(buf)), best, -1):
            if buf.endswith(m[:k]):
                best = k
                break
    return best


def _json_body(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


class ToolStreamer:
    """Incremental native-XML tool-call parser.

    Emits OpenAI tool-call argument deltas while preserving string payloads.
    Only the native template framing newline is removed on either side;
    strings remain text even when they look like JSON literals. Non-string
    values are buffered and coerced whole. Truncated calls are closed as
    valid JSON and marked so the HTTP response can report the token limit.
    """

    def __init__(self, tools=None):
        self.types = _tool_param_types(tools)
        self.buf = ""
        self.state = "idle"        # idle | block | func | val
        self.n = 0
        self.name = ""
        self.first_param = True
        self.mode = ""             # val: lead | str | raw
        self.val = ""              # buffered non-string value
        self.pend = ""
        self.truncated = False
        self.record = []           # [name, arguments] per call, as streamed

    def _flush(self, out):
        if self.pend:
            out.append({"index": self.n - 1, "function": {"arguments": self.pend}})
            self.record[-1][1] += self.pend
            self.pend = ""


    def _end_value(self, final_text: str):
        """The value's text up to its terminator (or the end of generation) is final_text."""
        if self.mode in ("lead", "str"):
            if final_text.endswith("\r\n"):
                final_text = final_text[:-2]
            elif final_text.endswith("\n"):
                final_text = final_text[:-1]
            self.pend += _json_body(final_text) + '"'
        else:
            v = coerce(self.val + final_text, self.ptype)
            self.pend += json.dumps(v, ensure_ascii=False)

    def feed(self, text: str, final: bool = False) -> list:
        out = []
        # Protocol-looking text inside a string argument can be literal code.
        # The backend stops on actual EOS token IDs; do not scrub payload bytes.
        self.buf += text
        while True:
            if self.state == "idle":
                i = self.buf.find(TOOL_OPEN)
                if i < 0:
                    h = _partial_suffix(self.buf, [TOOL_OPEN])
                    self.buf = self.buf[len(self.buf) - h:]
                    break
                self.buf, self.state = self.buf[i + len(TOOL_OPEN):], "block"
            elif self.state == "block":
                j, k = self.buf.find("<function="), self.buf.find("</tool_call>")
                if k >= 0 and (j < 0 or k < j):
                    self.buf, self.state = self.buf[k + len("</tool_call>"):], "idle"
                    continue
                if j < 0:
                    h = _partial_suffix(self.buf, ["<function=", "</tool_call>"])
                    self.buf = self.buf[len(self.buf) - h:]
                    break
                rest = self.buf[j + len("<function="):]
                g = rest.find(">")
                if g < 0:
                    self.buf = self.buf[j:]
                    break
                name = rest[:g]
                if not name or any(c.isspace() for c in name):
                    self.buf = rest            # not a function tag; the regex would look further on
                    continue
                self.name, self.buf = name, rest[g + 1:]
                out.append({"index": self.n, "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                            "function": {"name": self.name, "arguments": ""}})
                self.n += 1
                self.record.append([self.name, ""])
                self.pend, self.first_param, self.state = "{", True, "func"
            elif self.state == "func":
                cands = [(self.buf.find(t), t) for t in ("<parameter=", "</function>", "</tool_call>")]
                cands = [c for c in cands if c[0] >= 0]
                if not cands:
                    h = _partial_suffix(self.buf, ["<parameter=", "</function>", "</tool_call>"])
                    self.buf = self.buf[len(self.buf) - h:]
                    break
                at, tag = min(cands)
                if tag != "<parameter=":       # the call is complete (</tool_call> first: malformed, close it)
                    self.pend += "}"
                    self._flush(out)
                    self.buf = self.buf[at + len(tag):]
                    self.state = "block" if tag == "</function>" else "idle"
                    continue
                rest = self.buf[at + len("<parameter="):]
                g = rest.find(">")
                if g < 0:
                    self.buf = self.buf[at:]
                    break
                pname = rest[:g]
                if not pname or any(c.isspace() for c in pname):
                    self.buf = rest
                    continue
                self.buf = rest[g + 1:]
                self.ptype = self.types.get(self.name, {}).get(pname)
                self.pend += ("" if self.first_param else ", ") + json.dumps(pname, ensure_ascii=False) + ": "
                self.first_param = False
                self.val = ""
                if self.ptype == "string":
                    self.pend += '"'
                    self.mode = "lead"
                else:
                    self.mode = "raw"
                self.state = "val"
            else:                                   # val
                if self.mode == "lead":
                    if not self.buf:
                        break
                    if self.buf == "\r" and not final:
                        break
                    if self.buf.startswith("\r\n"):
                        self.buf = self.buf[2:]
                    elif self.buf.startswith("\n"):
                        self.buf = self.buf[1:]
                    self.mode = "str"
                end = self.buf.find("</parameter>")
                if end >= 0:
                    self._end_value(self.buf[:end])
                    self.buf = self.buf[end + len("</parameter>"):]
                    self.state = "func"
                    self._flush(out)
                    continue
                h = _partial_suffix(self.buf, ["</parameter>"])
                avail, keep = self.buf[:len(self.buf) - h], self.buf[len(self.buf) - h:]
                if self.mode == "str":
                    # Retain at most the possible CRLF framing newline. All
                    # indentation, spaces and additional newlines are payload.
                    k = max(0, len(avail) - 2)
                    self.pend += _json_body(avail[:k])
                    self.buf = avail[k:] + keep
                else:
                    self.val += avail
                    self.buf = keep
                break
        if final:
            self._finish(out)
        else:
            self._flush(out)
        return out

    def _finish(self, out):
        if self.state == "val":
            self.truncated = True
            self._end_value(self.buf)
            self.buf = ""
            self.state = "func"
        if self.state == "func":
            self.truncated = True
            self.pend += "}"
            self.state = "idle"
        self._flush(out)


KEEPALIVE_S = float(os.environ.get("SSE_KEEPALIVE_S", "2"))


def _check_streamed(streamer, calls):
    """Log (never alter: it is already sent) any difference between the streamed calls and the
    one-shot parse. Names, count and parsed arguments are compared."""
    got = getattr(streamer, "record", None)
    if got is None:
        return
    want = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
    have = []
    for name, args in got:
        try:
            have.append((name, json.loads(args)))
        except Exception:
            have.append((name, args))
    if have != want:
        log(f"tool stream differs from one-shot parse: streamed {len(have)} call(s), parsed {len(want)}"
            + (" (truncated)" if streamer.truncated else ""))


class Splitter:
    """Turn a raw delta stream into (reasoning, content) deltas.

    This pack's template opens the thinking block itself, so the model streams
    reasoning first with no opening tag and closes it with `</think>`. Default to
    reasoning (unless start_in_content=True when the prompt already closed
    `</think>`), switch to content at the close tag, and stop emitting once a
    `<tool_call>` appears (the rest is parsed into tool_calls at the end).
    """

    TAIL = 24  # covers both <｜end▁of▁thinking｜> (19 chars) and <tool_call> (11 chars)

    def __init__(self, start_in_content: bool = False):
        self.raw = ""
        self.phase = "content" if start_in_content else "reasoning"
        self.sent_reason = 0
        self.content_start = 0 if start_in_content else None
        self.sent_content = 0
        self._stripped_open = False
        self._pending_open_newline = False

    def _enter_tool(self):
        self.phase = "tool"

    def feed(self, delta: str):
        self.raw += delta
        if self._pending_open_newline and self.raw:
            if self.raw.startswith("\n"):
                self.raw = self.raw[1:]
            self._pending_open_newline = False
        if not self._stripped_open:
            stripped = self.raw.lstrip()
            for otag in OPEN_TAGS:
                if stripped.startswith(otag):
                    idx = self.raw.find(otag) + len(otag)
                    self._pending_open_newline = idx == len(self.raw)
                    if idx < len(self.raw) and self.raw[idx] == "\n":
                        idx += 1
                    self.raw = self.raw[idx:]
                    self._stripped_open = True
                    break
            else:
                if len(stripped) >= max(len(t) for t in OPEN_TAGS):
                    self._stripped_open = True
        if self.phase == "tool":
            return [("tool", delta)] if delta else []
        out = []

        # Resolve the reasoning boundary first, including when the same delta
        # also completes a tool marker. Otherwise reasoning and its close tag
        # are mistakenly emitted as visible content.
        if self.phase == "reasoning":
            close = -1
            for tag in CLOSE_TAGS:
                i = self.raw.find(tag)
                if i >= 0 and (close < 0 or i < close):
                    close = i
            if close >= 0:
                if close > self.sent_reason:
                    out.append(("reasoning", self.raw[self.sent_reason:close]))
                self.sent_reason = close
                for tag in CLOSE_TAGS:
                    if self.raw.startswith(tag, close):
                        self.content_start = close + len(tag)
                        break
                self.phase = "content"
            elif TOOL_OPEN not in self.raw:
                safe = max(0, len(self.raw) - self.TAIL)
                if safe > self.sent_reason:
                    out.append(("reasoning", self.raw[self.sent_reason:safe]))
                    self.sent_reason = safe
                return out
            else:
                # Some turns go straight from thinking to a tool marker.
                # Preserve the preceding thought in the reasoning channel.
                cut = self.raw.find(TOOL_OPEN)
                if cut > self.sent_reason:
                    out.append(("reasoning", self.raw[self.sent_reason:cut]))
                self.sent_reason = cut
                self.content_start = cut
                self.phase = "content"

        start = self.content_start if self.content_start is not None else 0
        cut = self.raw.find(TOOL_OPEN, start)
        if cut >= 0:
            if cut > start + self.sent_content:
                out.append(("content", self.raw[start + self.sent_content:cut]))
                self.sent_content = cut - start
            self._enter_tool()
            out.append(("tool", self.raw[cut:]))      # v4: the call streams (ToolStreamer)
            return out

        if self.phase == "content":
            # Hold a marker-sized tail until the next delta or finish().
            limit = max(start, len(self.raw) - self.TAIL)
            if limit > start + self.sent_content:
                out.append(("content", self.raw[start + self.sent_content:limit]))
                self.sent_content = limit - start
        return out

    def combined(self):
        """Everything after the thinking block: the answer and any tool markup."""
        return self.raw[self.content_start:] if self.content_start is not None else ""

    def finish(self):
        out = []
        if self.phase == "reasoning":
            tail = self.raw[self.sent_reason:]
            for tag in CLOSE_TAGS:
                tail = tail.split(tag)[0]
            # Exhausting the budget inside thinking does not make its tail an
            # answer. Keep both response modes consistent with the emitted deltas.
            if tail:
                out.append(("reasoning", tail))
                self.sent_reason = len(self.raw)
        elif self.phase == "content":
            start = self.content_start
            tool_start = self.raw.find(TOOL_OPEN, start)
            limit = tool_start if tool_start >= 0 else len(self.raw)
            if limit > start + self.sent_content:
                out.append(("content", self.raw[start + self.sent_content:limit]))
        return out


class Engine:
    """Serving-protocol engine bound to a generation backend.

    The real backend is exllamav3 (see load_real_engine); tests bind explicit
    fakes with the same surface: tokenizer.encode/single_id/hf_render_chat_template,
    generator.enqueue/iterate/cancel, and a Job factory with the pinned Job kwargs.
    """

    def __init__(self, tok, gen, backend, ctx, ndt, served_name, spec=None,
                 guard_default=None, sampling_defaults=None, thinking_default=True,
                 default_max_tokens=DEFAULT_MAX_TOKENS):
        self.tok = tok
        self.gen = gen
        self.backend = backend
        self.ndt = ndt
        self.ctx = ctx
        self.name = served_name
        self.lock = threading.Lock()
        self.runtime_info = getattr(backend, "runtime_info", lambda: {})()
        # Fleet default for the repetition guard (REPETITION_GUARD, read once at startup).
        # None means a request is guarded only if it asks; a request that carries the field
        # always wins, including an explicit false that switches it off for that request.
        self.guard_default = guard_default
        self.sampling_defaults = sampling_defaults or {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
        self.thinking_default = thinking_default
        self.default_max_tokens = default_max_tokens

        # Speculative-decoding mode label. AR (ndt == 0) and MTP2 (ndt == 2)
        # keep the exact labels inferred from ndt as before; the DFlash2
        # draft-model mode passes its own. spec_label feeds /health and the
        # ready log; props_spec preserves /props's pre-existing value for AR
        # mode, which reports the ndt integer 0 rather than "off".
        self.spec_label = spec or ("mtp" if ndt else "off")
        self.props_spec = spec or (ndt and "mtp")

        self.stop_ids = []
        for name in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = self.tok.single_id(name)
                if isinstance(tid, int):
                    self.stop_ids.append(tid)
            except Exception:
                pass
        if not self.stop_ids:
            self.stop_ids = None
        self._jinja_tpl = None
        self._jinja_meta = {}
        log(f"ready: {served_name} ctx={ctx} spec={self.spec_label}")

    @staticmethod
    def normalise(messages):
        """Make client history acceptable to the pack's template.

        OpenAI sends `tool_calls[].function.arguments` as a JSON string; the
        template iterates it as a mapping (`arguments|items`), which raised
        "Can only get item pairs from a mapping" and pushed every follow-up
        render into the crude fallback (no tools, no open thinking block).
        """
        out = []
        system = []
        for m in messages or []:
            m = dict(m or {})
            role = m.get("role")
            if role not in ("system", "developer", "user", "assistant", "tool"):
                raise RequestError(f"Unsupported message role: {role!r}")
            content = m.get("content")
            if isinstance(content, list):
                # This endpoint loads a text-only model. Never silently turn an
                # image or an unknown part into a string and change the prompt.
                if any(not isinstance(p, dict) or p.get("type") != "text"
                       or not isinstance(p.get("text"), str) for p in content):
                    raise RequestError("Only text content parts are supported by this model")
                m["content"] = "\n".join(p["text"] for p in content)
            elif content is not None and not isinstance(content, str):
                raise RequestError("Message content must be text, text parts, or null")
            if role in ("system", "developer"):
                # Qwen's template accepts one leading system message. Preserve
                # all client instructions in order instead of losing the tool
                # schema and history through a generic-template fallback.
                if m.get("content"):
                    system.append(m["content"])
                continue
            tcs = m.get("tool_calls")
            if isinstance(tcs, list):
                fixed = []
                for tc in tcs:
                    tc = dict(tc or {})
                    fn = dict(tc.get("function") or {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args) if args.strip() else {}
                        except (ValueError, TypeError) as exc:
                            raise RequestError("Tool-call arguments must be a JSON object") from exc
                    if not isinstance(fn.get("arguments", {}), dict):
                        raise RequestError("Tool-call arguments must be a JSON object")
                    tc["function"] = fn
                    fixed.append(tc)
                m["tool_calls"] = fixed
            out.append(m)
        if system:
            out.insert(0, {"role": "system", "content": "\n\n".join(system)})
        return out

    def _render_jinja_fallback(self, messages, **kwargs):
        if self._jinja_tpl is None:
            import jinja2
            from jinja2.ext import loopcontrols
            cfg = getattr(self.tok, "config", None)
            mdir = getattr(cfg, "directory", None)
            if not mdir:
                raise RuntimeError("No tokenizer config.directory available for jinja2 chat template")
            tc_path = os.path.join(mdir, "tokenizer_config.json")
            tc = {}
            if os.path.isfile(tc_path):
                with open(tc_path, encoding="utf-8") as handle:
                    tc = json.load(handle)
            jinja_path = os.path.join(mdir, "chat_template.jinja")
            tpl_str = tc.get("chat_template")
            if os.path.isfile(jinja_path):
                with open(jinja_path, encoding="utf-8") as handle:
                    tpl_str = handle.read()
            if not tpl_str:
                raise RuntimeError(f"No chat_template found in {mdir}")
            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
            env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
            def raise_exception(message):
                raise RequestError(str(message))
            env.globals["raise_exception"] = raise_exception
            self._jinja_tpl = env.from_string(tpl_str)
            self._jinja_meta = {
                "bos_token": tc.get("bos_token") or "",
                "eos_token": tc.get("eos_token") or "<|im_end|>",
            }
        return self._jinja_tpl.render(
            messages=messages,
            add_generation_prompt=True,
            **self._jinja_meta,
            **kwargs,
        )

    def render(self, messages, tools=None, enable_thinking=None):
        kwargs = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        if tools:
            kwargs["tools"] = tools
        messages = self.normalise(messages)
        try:
            return self.tok.hf_render_chat_template(messages, add_generation_prompt=True, **kwargs)
        except ModuleNotFoundError:
            pass
        except Exception as exc:
            log("using native Jinja chat template:", type(exc).__name__, str(exc)[:160])
        try:
            return self._render_jinja_fallback(messages, **kwargs)
        except RequestError:
            raise
        except Exception as exc:
            raise RequestError(f"Cannot render the model chat template: {exc}") from exc

    def encode(self, prompt_text):
        """Tokenize the rendered prompt once; returns (ids, n_prompt)."""
        ids = self.tok.encode(prompt_text, encode_special_tokens=True)
        n_prompt = int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)
        return ids, n_prompt

    def resolve_budget(self, requested, n_prompt):
        """Effective max_new_tokens, bounded by ctx and the output ceiling.

        A missing budget falls back to DEFAULT_MAX_TOKENS clamped to the room
        left in ctx. An explicit budget beyond either limit raises BudgetError so the
        client gets a 400 before any SSE headers are sent. The returned value is
        exactly what flows to Job.max_new_tokens on the generator API.
        """
        room = self.ctx - n_prompt
        if room < 1:
            raise BudgetError(
                f"This model's maximum context length is {self.ctx} tokens but the "
                f"prompt alone is {n_prompt} tokens; reduce the prompt")
        if requested is None:
            return min(self.default_max_tokens, room)
        if requested > room:
            raise BudgetError(
                f"This model's maximum context length is {self.ctx} tokens. However, "
                f"you requested {n_prompt + requested} tokens ({n_prompt} in the "
                f"messages, {requested} in the completion). Please reduce the length "
                f"of the messages or completion.")
        if requested > DEFAULT_MAX_TOKENS:
            raise BudgetError(
                f"max_tokens must be <= {DEFAULT_MAX_TOKENS} for this model "
                f"(output ceiling); you requested {requested}")
        return requested

    def stream(self, prompt_text, max_new, sampler, ids=None, n_prompt=None, guard=None,
               seed=None, metrics=None, cancel_event=None):
        """Yield (kind, text) chunks: kind in {"reasoning", "content"}.

        max_new is the RESOLVED per-request budget (see resolve_budget) and is
        forwarded verbatim as Job.max_new_tokens. Passing ids/n_prompt skips the
        redundant re-encode when the caller already tokenized the prompt.

        Closing this generator (GeneratorExit, e.g. the handler's gen.close()
        after a client disconnect) cancels the backend job and releases the
        engine lock instead of leaving the job running with nobody reading it.

        guard is the validated per-request repetition_guard (None = off, True = the
        engine defaults, dict = options). It is passed to Job ONLY when the client
        opted in, so the Job kwargs of an unopted request are exactly the v2 kwargs.
        The guard summary the engine reports on the final result, when it reports one,
        is returned as the fifth element.

        Returns via StopIteration: (splitter, eos_reason, n_prompt, n_out, guard_summary).
        """
        if ids is None or n_prompt is None:
            ids, n_prompt = self.encode(prompt_text)

        with self.lock:
            if cancel_event is not None and cancel_event.is_set():
                raise ConnectionResetError("Client disconnected before generation")
            job_kwargs = dict(
                input_ids=ids,
                max_new_tokens=max_new,
                min_new_tokens=1,
                stop_conditions=self.stop_ids,
                sampler=sampler,
                decode_special_tokens=True,
            )
            if guard is not None:
                job_kwargs["repetition_guard"] = guard
            if seed is not None:
                job_kwargs["seed"] = seed
            job = self.backend.Job(**job_kwargs)
            guard_summary = None
            serial = self.gen.enqueue(job)
            start_in_content = prompt_text.rstrip().endswith(CLOSE_TAGS)
            splitter = Splitter(start_in_content=start_in_content)
            eos_reason = "stop_token"
            done = False
            try:
                while not done:
                    if cancel_event is not None and cancel_event.is_set():
                        raise ConnectionResetError("Client disconnected during generation")
                    for r in self.gen.iterate():
                        if r.get("serial") != serial:
                            continue
                        if r.get("stage") == "error":
                            raise r["error"]
                        if r.get("stage") == "streaming":
                            piece = r.get("text") or ""
                            if piece:
                                for kind, text in splitter.feed(piece):
                                    yield kind, text
                        if r.get("eos"):
                            if metrics is not None:
                                metrics.update({k: r[k] for k in (
                                    "time_prefill", "time_generate", "cached_tokens",
                                    "new_tokens", "accepted_draft_tokens", "rejected_draft_tokens"
                                ) if k in r})
                            eos_reason = r.get("eos_reason") or "stop_token"
                            guard_summary = r.get("repetition_guard")
                            done = True
                for kind, text in splitter.finish():
                    yield kind, text
            finally:
                # Runs on normal completion (job finished; cancel is then a
                # no-op), on an error-stage exception, and on GeneratorExit
                # from a client disconnect: never leave a job on the backend
                # that nobody is reading.
                if not done:
                    try:
                        self.gen.cancel(job)
                    except Exception as exc:
                        log("job cancel failed:", type(exc).__name__, str(exc)[:120])

        ids_out = self.tok.encode(splitter.raw)
        n_out = int(ids_out.shape[-1]) if hasattr(ids_out, "shape") else len(ids_out)
        return splitter, eos_reason, n_prompt, n_out, guard_summary


def load_real_engine(model_dir, ctx, ndt, served_name, draft_model_dir=None,
                     guard_default=None, chunk_size=2048, thinking_default=True,
                     default_max_tokens=32768, recurrent_state_dtype="fp32"):
    """Bind an Engine to the real exllamav3 backend (mirrors ornith122-serve.py).

    Imports are lazy so importing this module never pulls torch. Signatures match
    the pinned exllamav3 source: Generator(model, cache, tok, draft_model=,
    draft_cache=, num_draft_tokens=), gen.enqueue(Job)->serial,
    gen.iterate()->[{stage,text,eos,eos_reason,serial}], gen.cancel(job) (the
    Job object, not the serial), Job(input_ids=, max_new_tokens=,
    min_new_tokens=, stop_conditions=, sampler=, decode_special_tokens=),
    exllamav3.generator.sampler.ArgmaxSampler() /
    TopPSampler(top_p=, temperature=, temperature_last=).

    draft_model_dir selects the opt-in native DFlash2 mode (staged HF/BF16
    drafter, no conversion): the drafter is loaded from its own directory, its
    caps decide the draft depth (default_draft_size = block_size - 1 = 15) and
    the target cache history, and the Generator binds it via attach_to inside
    its own __init__. The directory is verified to be a native DFlash2 draft
    model (architecture DFlash2DraftModel plus the dflash_draft and
    default_draft_size caps) before any cache is sized; anything else exits
    without loading. Without it, ndt 0/2 keep the AR / target-dir MTP paths
    exactly as before.
    """
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer, Job as _Job
    from exllamav3.generator import sampler as sampler_mod
    import torch
    if recurrent_state_dtype not in ("fp16", "fp32"):
        raise ValueError("recurrent_state_dtype must be fp16 or fp32")
    state_dtype = torch.float32 if recurrent_state_dtype == "fp32" else torch.float16

    class _Backend:
        Job = _Job
        ArgmaxSampler = sampler_mod.ArgmaxSampler
        TopPSampler = sampler_mod.TopPSampler

        @staticmethod
        def make_sampler(temperature, top_p, top_k, min_p, presence_penalty,
                         frequency_penalty, repetition_penalty):
            steps = []
            if repetition_penalty != 1.0:
                steps.append(sampler_mod.SS_RepP(repetition_penalty))
            if presence_penalty or frequency_penalty:
                steps.append(sampler_mod.SS_PresFreqP(presence_penalty, frequency_penalty))
            if temperature <= 0:
                if not steps:
                    return sampler_mod.ArgmaxSampler()
                steps.append(sampler_mod.SS_Argmax())
            else:
                steps.extend([sampler_mod.SS_Temperature(temperature),
                              sampler_mod.SS_TopK(top_k), sampler_mod.SS_TopP(top_p),
                              sampler_mod.SS_MinP(min_p), sampler_mod.SS_Sample()])
            return sampler_mod.CustomSampler(steps)

        @staticmethod
        def runtime_info():
            import torch
            return {"torch": torch.__version__, "hip": torch.version.hip,
                    "recurrent_state_dtype": recurrent_state_dtype}

    draft = draft_cache = None
    spec = None
    if draft_model_dir:
        # Build the DFlash2 drafter first: its caps (default_draft_size = 15
        # for the block-16 drafter) size the target cache below, which must
        # exist before either model loads. No weights move until load().
        draft_config = Config.from_directory(draft_model_dir)
        draft = Model.from_config(draft_config)
        # Verify the declared mode before sizing anything: only a native
        # DFlash2 draft model may run as "dflash2". v1 DFlash and Laguna
        # directories carry the same dflash caps under a different
        # architecture and would silently run mislabeled; the target dir
        # itself would load 122B twice under a false label. Rejected here,
        # before any cache exists and before anything loads.
        if (draft_config.architecture != "DFlash2DraftModel"
                or not draft.caps.get("dflash_draft")
                or "default_draft_size" not in draft.caps):
            raise SystemExit(
                f"--draft-model: {draft_model_dir!r} is not a native DFlash2 "
                f"draft model (architecture {draft_config.architecture!r}); "
                f"refusing to size caches or load it as dflash2")
        ndt = draft.caps["default_draft_size"]
        spec = "dflash2"

    t0 = time.time()
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=ctx, max_history=max(3, ndt or 0), max_batch_size=1,
                  recurrent_dtype=state_dtype)
    tok = Tokenizer.from_config(config)
    model.load()
    log(f"target loaded in {time.time() - t0:.1f}s arch={config.architecture}")

    if spec:
        t0 = time.time()
        # One recurrent draft slot: the engine lock serializes jobs, and the
        # Cache default of 16 slots would pin ~960 MiB of SWA rings for this
        # 60 MiB-per-slot drafter.
        draft_cache = Cache(draft, max_num_tokens=ctx, max_batch_size=1,
                            max_history=max(3, ndt), recurrent_dtype=state_dtype)
        draft.load()
        log(f"DFlash2 drafter loaded in {time.time() - t0:.1f}s ndt={ndt}")
    elif ndt:
        t0 = time.time()
        draft = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft, max_num_tokens=ctx, max_history=max(3, ndt), max_batch_size=1,
                            recurrent_dtype=state_dtype)
        draft.load()
        log(f"MTP drafter loaded in {time.time() - t0:.1f}s ndt={ndt}")

    kw = {}
    if draft is not None:
        kw = dict(draft_model=draft, draft_cache=draft_cache, num_draft_tokens=ndt)
    gen = Generator(model, cache, tok, max_batch_size=1, max_chunk_size=chunk_size, **kw)
    ndt = gen.num_draft_tokens
    generation_path = os.path.join(model_dir, "generation_config.json")
    generation = {}
    if os.path.isfile(generation_path):
        with open(generation_path, encoding="utf-8") as handle:
            generation = json.load(handle)
    defaults = {"temperature": generation.get("temperature", 1.0),
                "top_p": generation.get("top_p", 1.0),
                "top_k": generation.get("top_k", 0)}
    if generation.get("do_sample") is False:
        defaults["temperature"] = 0.0
    return Engine(tok, gen, _Backend(), ctx, ndt, served_name, spec=spec,
                  guard_default=guard_default, sampling_defaults=defaults,
                  thinking_default=thinking_default, default_max_tokens=default_max_tokens)


def generation_timings(metrics, n_prompt, n_out, elapsed):
    cached = int(metrics.get("cached_tokens", 0))
    prefill = float(metrics.get("time_prefill", 0.0))
    decode = float(metrics.get("time_generate", elapsed))
    result = {"prompt_n": max(0, n_prompt - cached), "prompt_cached_n": cached,
              "prompt_ms": prefill * 1000, "predicted_n": n_out,
              "predicted_ms": decode * 1000, "total_ms": elapsed * 1000,
              "prompt_per_second": max(0, n_prompt - cached) / prefill if prefill > 0 else 0.0,
              "predicted_per_second": n_out / max(decode, .001)}
    for key in ("accepted_draft_tokens", "rejected_draft_tokens"):
        if key in metrics:
            result[key] = metrics[key]
    return result


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def _json(self, code, obj):
            payload = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._json(200, {"status": "ok", "model": engine.name, "context": engine.ctx,
                                 "speculative": engine.spec_label,
                                 "tools": True, "streaming": True})
            elif self.path.startswith("/props"):
                self._json(200, {"model": engine.name, "context": engine.ctx,
                                 "speculative": engine.props_spec, "num_draft_tokens": engine.ndt,
                                 "tools": True, "streaming": True,
                                 "default_max_tokens": engine.default_max_tokens,
                                 "max_tokens_limit": DEFAULT_MAX_TOKENS,
                                 "sampling_defaults": engine.sampling_defaults,
                                 "thinking_default": engine.thinking_default,
                                 "prefill_chunk": getattr(engine.gen, "max_chunk_size", None),
                                 "runtime": engine.runtime_info,
                                 **({"repetition_guard_default": engine.guard_default}
                                    if engine.guard_default is not None else {})})
            elif self.path.startswith("/v1/models"):
                self._json(200, {"object": "list", "data": [{
                    "id": engine.name, "object": "model", "created": int(time.time()),
                    "owned_by": "exllamav3", "context_length": engine.ctx}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.path.startswith(("/v1/chat/completions", "/v1/completions")):
                self._json(404, {"error": {"message": "not found"}})
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._json(400, {"error": {"message": "bad json"}})
                return
            if not isinstance(body, dict):
                self._json(400, {"error": {"message": "request body must be a JSON object",
                                            "type": "invalid_request_error"}})
                return
            if self.path.startswith("/v1/completions"):
                body = dict(body)
                body["messages"] = [{"role": "user", "content": body.get("prompt") or ""}]

            sse_started = False
            try:
                messages = body.get("messages")
                if (not isinstance(messages, list) or not messages
                        or not all(isinstance(m, dict) for m in messages)):
                    raise RequestError("messages must be a non-empty list of message objects")
                max_requested = _int_param(body, "max_tokens", minimum=1)
                defaults = engine.sampling_defaults
                temp = _float_param(body, "temperature", 0.0, 2.0, default=defaults["temperature"])
                top_p = _float_param(body, "top_p", 0.0, 1.0, default=defaults["top_p"], lo_exclusive=True)
                top_k = _int_param(body, "top_k", minimum=0)
                if top_k is None:
                    top_k = defaults["top_k"]
                min_p = _float_param(body, "min_p", 0.0, 1.0, default=0.0)
                presence = _float_param(body, "presence_penalty", -2.0, 2.0, default=0.0)
                frequency = _float_param(body, "frequency_penalty", -2.0, 2.0, default=0.0)
                repetition = _float_param(body, "repetition_penalty", 0.0, 10.0, default=1.0, lo_exclusive=True)
                seed = _int_param(body, "seed", minimum=0)
                guard = _guard_param(body)
                guard_explicit = guard is not _ABSENT
                if not guard_explicit:
                    guard = engine.guard_default

                tools = body.get("tools")
                if tools is not None:
                    if (not isinstance(tools, list)
                            or any(not isinstance(t, dict) for t in tools)):
                        raise RequestError("tools must be a list of function objects")
                    if any("function" in t and not isinstance(t["function"], dict) for t in tools):
                        raise RequestError("tool function must be an object")
                    # The native template expects the OpenAI tool definitions.
                    # Parsing accepts either envelope through _tool_param_types;
                    # stripping it here silently changes the model's prompt.
                ctk = body.get("chat_template_kwargs")
                if ctk is None:
                    ctk = {}
                if not isinstance(ctk, dict):
                    raise RequestError("chat_template_kwargs must be an object")
                thinking = ctk.get("enable_thinking", body.get("enable_thinking"))
                if thinking is not None and not isinstance(thinking, bool):
                    raise RequestError("enable_thinking must be a boolean")
                if thinking is None:
                    thinking = engine.thinking_default
                prompt_text = engine.render(messages, tools=tools, enable_thinking=thinking)

                # Resolve the budget BEFORE choosing the response shape, so an
                # invalid or overflowing request gets a clean 400 even when it
                # asked for a stream.
                ids, n_prompt = engine.encode(prompt_text)
                max_new = engine.resolve_budget(max_requested, n_prompt)
                if hasattr(engine.backend, "make_sampler"):
                    sampler = engine.backend.make_sampler(temperature=temp, top_p=top_p,
                        top_k=top_k, min_p=min_p, presence_penalty=presence,
                        frequency_penalty=frequency, repetition_penalty=repetition)
                else:
                    sampler = (engine.backend.ArgmaxSampler() if temp <= 0
                               else engine.backend.TopPSampler(top_p=top_p, temperature=temp))
            except RequestError as exc:
                self._json(400, {"error": {"message": str(exc),
                                            "type": "invalid_request_error"}})
                return
            except BudgetError as exc:
                self._json(400, {"error": {"message": str(exc),
                                            "type": "invalid_request_error"}})
                return
            except Exception as exc:
                self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                return

            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            streaming = bool(body.get("stream"))
            t0 = time.time()
            metrics = {}

            try:
                if not streaming:
                    kinds = {"reasoning": "", "content": "", "tool": ""}
                    splitter = eos_reason = None
                    gen = engine.stream(prompt_text, max_new, sampler, ids=ids, n_prompt=n_prompt,
                                        guard=guard, seed=seed, metrics=metrics)
                    try:
                        while True:
                            try:
                                kind, text = next(gen)
                            except StopIteration as stop:
                                splitter, eos_reason, n_prompt, n_out, guard_summary = stop.value
                                guard_summary = _report_guard(guard_summary, guard_explicit)
                                break
                            kinds[kind] += text
                    finally:
                        gen.close()
                    content, calls = parse_tool_calls(splitter.combined(), tools=tools)
                    finish = "tool_calls" if calls else ("length" if eos_reason == "max_new_tokens" else "stop")
                    message = {"role": "assistant",
                               "content": None if calls else (content or "")}
                    if calls:
                        message["tool_calls"] = calls
                        if kinds["reasoning"]:
                            message["reasoning_content"] = kinds["reasoning"].strip()
                    elif kinds["reasoning"]:
                        message["reasoning_content"] = kinds["reasoning"].strip()
                    self._json(200, {
                        "id": cid, "object": "chat.completion", "created": created,
                        "model": engine.name,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                                  "total_tokens": n_prompt + n_out},
                        "timings": generation_timings(metrics, n_prompt, n_out, time.time() - t0),
                        **({"repetition_guard": guard_summary}
                           if guard_summary is not None else {}),
                    })
                    log(f"chat: {n_prompt}+{n_out} tok in {time.time() - t0:.1f}s eos={eos_reason} calls={len(calls)}")
                    return

                # ---- streaming ----
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                sse_started = True

                wlock = threading.Lock()
                last_write = [time.time()]
                alive = threading.Event()
                cancelled = threading.Event()

                def write(data: bytes):
                    with wlock:
                        self.wfile.write(data)
                        self.wfile.flush()
                        last_write[0] = time.time()

                def emit(delta, finish=None, usage=None, guard_summary=None):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                             "model": engine.name,
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if usage is not None:
                        chunk["usage"] = usage
                        chunk["timings"] = generation_timings(metrics, n_prompt, n_out, time.time() - t0)
                    if guard_summary is not None:
                        chunk["repetition_guard"] = guard_summary
                    write(f"data: {json.dumps(chunk)}\n\n".encode())

                def keepalive():
                    # v4: an SSE comment whenever nothing was written for KEEPALIVE_S (long prefill,
                    # or any quiet stretch), so clients and proxies never read silence as a hang
                    while not alive.wait(min(1.0, max(.05, KEEPALIVE_S))):
                        if time.time() - last_write[0] >= KEEPALIVE_S:
                            try:
                                write(b": keep-alive\n\n")
                            except Exception:
                                cancelled.set()
                                return

                ka = threading.Thread(target=keepalive, daemon=True)
                if KEEPALIVE_S > 0:
                    ka.start()
                streamer = ToolStreamer(tools)

                emit({"role": "assistant", "content": ""})
                collected = {"reasoning": "", "content": ""}
                splitter = eos_reason = None
                gen = engine.stream(prompt_text, max_new, sampler, ids=ids, n_prompt=n_prompt,
                                    guard=guard, seed=seed, metrics=metrics, cancel_event=cancelled)
                try:
                    while True:
                        try:
                            kind, text = next(gen)
                        except StopIteration as stop:
                            splitter, eos_reason, n_prompt, n_out, guard_summary = stop.value
                            guard_summary = _report_guard(guard_summary, guard_explicit)
                            break
                        if not text:
                            continue
                        if kind == "tool":
                            entries = streamer.feed(text)
                            if entries:
                                emit({"tool_calls": entries})
                            continue
                        collected[kind] += text
                        emit({"reasoning_content": text} if kind == "reasoning" else {"content": text})

                    entries = streamer.feed("", final=True)
                    if entries:
                        emit({"tool_calls": entries})
                    content, calls = parse_tool_calls(splitter.combined(), tools=tools)
                    if streamer.n == 0 and calls:      # nothing streamed (cannot happen normally): send whole
                        emit({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                    elif streamer.n:
                        _check_streamed(streamer, calls)
                    ncalls = streamer.n or len(calls)
                    if ncalls:
                        finish = "length" if streamer.truncated and eos_reason == "max_new_tokens" else "tool_calls"
                    else:
                        finish = "length" if eos_reason == "max_new_tokens" else "stop"
                    calls = [None] * ncalls
                    emit({}, finish, {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                                      "total_tokens": n_prompt + n_out},
                         guard_summary=guard_summary)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    log(f"stream: {n_prompt}+{n_out} tok in {time.time() - t0:.1f}s eos={eos_reason} calls={len(calls)}"
                        + (" truncated" if streamer.truncated else ""))
                finally:
                    alive.set()
                    # Deterministic cleanup: closing a still-suspended stream
                    # throws GeneratorExit inside Engine.stream, which cancels
                    # the backend job and frees the engine lock immediately
                    # instead of waiting for garbage collection.
                    gen.close()
            except (BrokenPipeError, ConnectionResetError):
                log("client disconnected mid-stream; backend job cancelled")
            except Exception as exc:
                log("generation failed:", type(exc).__name__, str(exc)[:200])
                if not sse_started:
                    try:
                        self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                    except Exception:
                        pass
                # After SSE headers were sent an HTTP error status can no longer
                # be delivered; terminate the event stream rather than splice a
                # status line into the SSE body.

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "0")))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("CTX", 8192)))
    ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("PREFILL_CHUNK", 2048)))
    ap.add_argument("--recurrent-state-dtype", choices=("fp16", "fp32"),
                    default=os.environ.get("RECURRENT_STATE_DTYPE", "fp32"))
    ap.add_argument("--thinking", choices=("on", "off"), default=os.environ.get("REASONING", "on"))
    ap.add_argument("--default-max-tokens", type=int, default=int(os.environ.get("RBUDGET", 32768)))
    # --draft 0 (AR) / --draft 2 (native MTP2, the default) and the opt-in
    # DFlash2 --draft-model are one mutually exclusive selection; argparse
    # rejects an explicit --draft together with --draft-model before loading.
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--draft", type=int,
                           default=int(os.environ.get("NUM_DRAFT", "2")))
    selection.add_argument("--draft-model", metavar="DIR",
                           help="opt-in native DFlash2 draft-model directory "
                                "(draft depth derived from the drafter's caps)")
    ap.add_argument("--served-name", default=os.environ.get("SERVED_NAME", "ornith-1.5-122b-exl3"))
    args = ap.parse_args()

    if not args.model:
        ap.error("--model or MODEL_DIR is required (no host-private default in the portable build)")
    if not 0 <= args.port <= 65535 or 18150 <= args.port <= 18159:
        ap.error("--port must be 0..65535, excluding the protected range 18150..18159")
    if args.ctx < 1:
        ap.error("--ctx must be positive")
    if not 0 <= args.draft <= 16:
        ap.error("--draft must be 0 (AR) or 1..16 (native MTP)")
    if args.chunk_size < 256 or args.chunk_size % 256:
        ap.error("--chunk-size must be a positive multiple of 256")
    if args.thinking not in ("on", "off"):
        ap.error("REASONING must be on or off")
    if args.recurrent_state_dtype not in ("fp16", "fp32"):
        ap.error("RECURRENT_STATE_DTYPE must be fp16 or fp32")
    if not 1 <= args.default_max_tokens <= DEFAULT_MAX_TOKENS:
        ap.error(f"--default-max-tokens / RBUDGET must be 1..{DEFAULT_MAX_TOKENS}")
    if args.draft_model is not None and not args.draft_model:
        ap.error("--draft-model must name a DFlash2 draft-model directory")
    if args.draft_model and args.draft == 0:
        # The mutually exclusive group catches an explicit --draft 0 together
        # with --draft-model; this catches the same contradiction arriving
        # through a NUM_DRAFT=0 environment default.
        ap.error("--draft-model (DFlash2) cannot be combined with --draft 0 (AR)")

    # The fleet default for the repetition guard, from REPETITION_GUARD. Read once at
    # startup: it decides only what a request that never mentions repetition_guard gets.
    guard_default = _guard_default()
    if guard_default is not None:
        log(f"repetition guard default: {guard_default!r}")
    if args.draft_model:
        engine = load_real_engine(args.model, args.ctx, args.draft, args.served_name,
                                  draft_model_dir=args.draft_model,
                                  guard_default=guard_default, chunk_size=args.chunk_size,
                                  thinking_default=args.thinking == "on",
                                  default_max_tokens=args.default_max_tokens,
                                  recurrent_state_dtype=args.recurrent_state_dtype)
    else:
        engine = load_real_engine(args.model, args.ctx, args.draft, args.served_name,
                                  guard_default=guard_default, chunk_size=args.chunk_size,
                                  thinking_default=args.thinking == "on",
                                  default_max_tokens=args.default_max_tokens,
                                  recurrent_state_dtype=args.recurrent_state_dtype)
    # The guard reports every detection and intervention on its own logger at INFO, and
    # Python discards INFO from a logger with no handler - which would make the documented
    # "shadow first" rollout impossible to read, since in shadow mode the response summary
    # only carries a count, not the period and copy length that size min_match/confirm.
    # Scoped to that one logger, so nothing else gets noisier and nothing is emitted unless
    # a request opted in.
    guard_log = logging.getLogger("exllamav3.generator.repetition_guard")
    guard_log.setLevel(logging.INFO)
    if not guard_log.handlers:
        guard_log.addHandler(logging.StreamHandler())
    guard_log.propagate = False

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    if 18150 <= srv.server_address[1] <= 18159:
        srv.server_close()
        ap.error("OS assigned a protected port; closed without serving")
    log(f"listening on http://{args.host}:{srv.server_address[1]}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
