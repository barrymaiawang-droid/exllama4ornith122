"""
Repetition guard: detect a job copying its own output and steer or stop it, invisibly to the client.

Tiers (Job(repetition_guard = {...}), off unless asked for):

    0  detect   the last `min_match` tokens equal an earlier span of this turn's own output, at distance
                `period`. That alone is NOT a loop: code, markup, tables and a draft rewritten after the
                thinking all repeat themselves. The copy is only tracked; output streams normally. Only
                in the last `hold` tokens before it could confirm is output held back (like a partial
                banned-string match), and it is released untouched the moment the copy breaks
    1  escape   a loop: the same block of `period` tokens occurs `repeats` times back to back, verbatim,
                AND that run is at least `confirm` tokens long. Roll the job back to the first held
                token and ban that token for one step
    2  nudge    a loop again within `escalate_within` tokens of tier 1: roll back again and inject the
                model-voiced interjection `nudge`, tagged hidden, so it steers the model but never reaches
                the client's text, token ids or completion
    3  stop     a loop after the nudge, or the `max_interventions` budget spent: end the job with
                eos_reason "repetition"

Why periodic and long: a one-time copy (a draft written out again, a quoted function) never overlaps its
own source, so its copy length stays below its period and it can never confirm. Healthy repetition with
a short period (closing tags, indentation, `0, 0, 0`) is broken by the next differing token long before
`confirm` tokens. Measured over ~5,900 real source/markup/data files tokenized with the Ornith tokenizer
(each written once and drafted-then-written): the previous rule (any 32-token copy, or one full period if
shorter) fired on 56-96% of files per type; this rule fired on one, a fuzz-test hex dump, while catching
100% of synthetic loops of period 1-400 within max(confirm, 3 * period) tokens.

Nothing streamed is ever retracted: a rollback discards held tokens only (at most `hold`). The detector matches token ids against the turn's own output only, so a model
quoting the prompt or a tool result never trips it. Shadow mode (mode = "shadow") detects and logs and
does nothing else: run it first to learn the loop lengths of real traffic before tuning the two numbers
that matter, `min_match` and `confirm`.

Inside a tool call or any constrained span (the job has filters) tier 2 is skipped: text must not be
injected into JSON. Every intervention is logged (logger "exllamav3.generator.repetition_guard", INFO;
the discarded text at DEBUG) and reported on the job's stream events and final result.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_NUDGE = "\n\nWait, I am repeating myself. Let me stop repeating and finish the answer differently.\n\n"

DEFAULTS = {
    "mode": "active",            # "active" | "shadow"
    "min_match": 8,              # tokens that must match an earlier span before a copy is tracked (tier 0)
    "confirm": 512,              # a loop's verbatim repeating run must be at least this many tokens (tier 1)
    "repeats": 4,                # ... and its block must occur this many times back to back (tier 1)
    "hold": 32,                  # output held (and discarded on a loop) before the run could confirm
    "max_interventions": 3,      # rollbacks (tier 1 / 2) per request before tier 3
    "escalate_within": 1024,     # a loop this soon after an intervention escalates to the next tier
    "nudge": DEFAULT_NUDGE,      # tier 2 text, hidden from the client; None or "" disables tier 2
}


@dataclass
class GuardEvent:
    kind: str                    # "none" | "suspect" | "extend" | "release" | "confirm"
    period: int = 0              # distance to the span being copied
    matched: int = 0             # length of the copy so far, counting from its first token
    hold_start: int = -1         # output index of the first held token (the one that completed the match)
    copy_start: int = -1         # output index where the copy began
    source_start: int = -1       # output index of the span being copied
    released: bool = False       # a copy in progress broke on this token (and a new one may start here)


class SuffixLoopDetector:
    """
    Token-level self-copy detector over one turn's output. feed() returns the event for the token just
    appended; truncate() drops tokens a rollback removed. O(1) per token: an index from every
    `min_match`-gram to the positions it ended at.
    """

    def __init__(self, min_match: int = 8, confirm: int = 512, repeats: int = 4, hold: int = 32):
        assert min_match >= 2, "min_match must be at least 2"
        assert confirm >= 1, "confirm must be at least 1"
        assert repeats >= 1, "repeats must be at least 1 (1: any copy of `confirm` tokens, no periodicity)"
        assert hold >= 0, "hold must be at least 0"
        self.min_match = min_match
        self.confirm = confirm
        self.repeats = repeats
        self.hold = hold
        self.tokens: list[int] = []
        self._index: dict[tuple, list[int]] = {}
        self.period = 0
        self.matched = 0
        self.hold_start = -1

    def __len__(self):
        return len(self.tokens)

    @property
    def suspecting(self) -> bool:
        return self.period > 0

    def confirm_length(self, period: int) -> int:
        """Copy length that counts as a loop: (repeats - 1) full periods copied, i.e. the block occurs
        `repeats` times back to back, and never shorter than `confirm` tokens"""
        return max(self.min_match, self.confirm, (self.repeats - 1) * period)

    def hold_length(self, period: int) -> int:
        """Copy length at which output starts being held: `hold` tokens before it could confirm"""
        return max(self.min_match, self.confirm_length(period) - self.hold)

    def _event(self, i: int, released: bool = False) -> GuardEvent:
        """Classify the copy in progress after token i; starts the hold when the copy gets close"""
        copy_start = i - self.matched + 1
        if self.matched >= self.confirm_length(self.period):
            if self.hold_start < 0:
                self.hold_start = i
            kind = "confirm"
        elif self.hold_start >= 0:
            kind = "extend"
        elif self.matched >= self.hold_length(self.period):
            self.hold_start = i
            kind = "suspect"
        else:
            kind = "release" if released else "none"
        return GuardEvent(kind, self.period, self.matched, self.hold_start, copy_start,
                          copy_start - self.period, released = released)

    def _gram(self, end: int) -> tuple | None:
        if end + 1 < self.min_match:
            return None
        return tuple(self.tokens[end + 1 - self.min_match : end + 1])

    def _register(self, end: int):
        g = self._gram(end)
        if g is not None:
            self._index.setdefault(g, []).append(end)

    def _reset(self):
        self.period = 0
        self.matched = 0
        self.hold_start = -1

    def feed(self, token: int) -> GuardEvent:
        self.tokens.append(token)
        i = len(self.tokens) - 1
        released = False

        if self.period:
            if token == self.tokens[i - self.period]:
                self.matched += 1
                self._register(i)
                return self._event(i)
            # a copy that was only tracked held nothing, so its end releases nothing
            released = self.hold_start >= 0
            self._reset()

        g = self._gram(i)
        earlier = self._index.get(g) if g is not None else None
        last_end = earlier[-1] if earlier else None
        self._register(i)
        if last_end is None:
            return GuardEvent("release" if released else "none", released = released)

        # The most recent earlier occurrence gives the period. The copy began at least min_match tokens
        # ago; measure how far back it really reaches (bounded by the confirm length, so O(confirm) once
        # per new copy), so a loop is not under-counted when the gram that reveals it arrives late
        period = i - last_end
        cap = self.confirm_length(period)
        matched = self.min_match
        while matched < cap and i - matched - period >= 0 and \
                self.tokens[i - matched] == self.tokens[i - matched - period]:
            matched += 1
        self.period = period
        self.matched = matched
        return self._event(i, released)

    def truncate(self, length: int):
        """Forget every token at output index >= length (a rollback), and any suspicion in progress"""
        assert 0 <= length <= len(self.tokens)
        for end in range(len(self.tokens) - 1, length - 1, -1):
            g = self._gram(end)
            if g is None:
                continue
            ends = self._index.get(g)
            if ends and ends[-1] == end:
                ends.pop()
                if not ends:
                    del self._index[g]
        del self.tokens[length:]
        self._reset()


class RepetitionGuard:
    """Per-job tier state over a SuffixLoopDetector; see the module docstring"""

    def __init__(self, options: dict | bool | None):
        opts = dict(DEFAULTS)
        if isinstance(options, dict):
            unknown = set(options) - set(DEFAULTS)
            assert not unknown, f"repetition_guard: unknown options {sorted(unknown)}"
            opts.update(options)
        assert opts["mode"] in ("active", "shadow"), "repetition_guard: mode must be 'active' or 'shadow'"
        assert opts["max_interventions"] >= 0 and opts["escalate_within"] >= 0
        self.options = opts
        self.mode = opts["mode"]
        self.detector = SuffixLoopDetector(opts["min_match"], opts["confirm"], opts["repeats"], opts["hold"])
        self.nudge_text = opts["nudge"] or None
        self._nudge_ids = None
        self.interventions: list[dict] = []
        self._quiet_until = -1
        self.detections = 0          # confirmed loops, acted on or not
        self.rollback_tokens = 0
        self.injected_tokens = 0

    @property
    def active(self) -> bool:
        return self.mode == "active"

    @property
    def position(self) -> int:
        """Output tokens seen so far (the next token's index)"""
        return len(self.detector)

    def feed(self, token: int) -> GuardEvent:
        ev = self.detector.feed(token)
        # Shadow mode never rolls back, so a loop would confirm again on every token; one note per
        # `confirm` tokens is the record, and the detector is re-armed from scratch each time
        if ev.kind == "confirm" and not self.active:
            if self.position <= self._quiet_until:
                ev = GuardEvent("none")
            else:
                self._quiet_until = self.position + self.options["confirm"]
        return ev

    def observe(self, event: GuardEvent):
        """A confirmed copy the guard will not act on (shadow mode, or a hold owned elsewhere): log and re-arm"""
        self.record(0, event)
        self.detector._reset()

    def truncate(self, length: int):
        self.detector.truncate(length)

    def plan(self, allow_nudge: bool) -> int:
        """Tier for a loop confirmed at the current position"""
        if len(self.interventions) >= self.options["max_interventions"]:
            return 3
        last = self.interventions[-1] if self.interventions else None
        if last is not None and self.position - last["position"] <= self.options["escalate_within"]:
            if last["tier"] == 1 and allow_nudge and self.nudge_text:
                return 2
            return 3
        return 1

    def nudge_ids(self, tokenizer):
        """The interjection as a (1, S) id tensor, tokenized once without special tokens"""
        if self._nudge_ids is None and self.nudge_text:
            ids = tokenizer.encode(self.nudge_text, add_bos = False, encode_special_tokens = False)
            self._nudge_ids = ids if ids.shape[-1] > 0 else None
        return self._nudge_ids

    def record(self, tier: int, event: GuardEvent, rollback_tokens: int = 0, injected_tokens: int = 0,
               discarded_text: str | None = None) -> dict:
        rec = {
            "tier": tier,
            "position": self.position,
            "period": event.period,
            "matched": event.matched,
            "copy_start": event.copy_start,
            "source_start": event.source_start,
            "rollback_tokens": rollback_tokens,
            "injected_tokens": injected_tokens,
        }
        self.detections += 1
        self.rollback_tokens += rollback_tokens
        self.injected_tokens += injected_tokens
        if tier > 0:
            self.interventions.append(rec)
        logger.info("repetition guard %s tier %d at output token %d: period %d, copy of %d tokens from %d, "
                    "rolled back %d, injected %d",
                    "shadow" if tier == 0 else "active", tier, rec["position"], event.period, event.matched,
                    event.source_start, rollback_tokens, injected_tokens)
        if discarded_text is not None and logger.isEnabledFor(logging.DEBUG):
            logger.debug("repetition guard discarded: %r", discarded_text)
        return rec

    def summary(self, visible_new_tokens: int | None = None) -> dict:
        s = {
            "mode": self.mode,
            "detections": self.detections,
            "interventions": list(self.interventions),
            "tier": max((r["tier"] for r in self.interventions), default = 0),
            "rollback_tokens": self.rollback_tokens,
            "injected_tokens": self.injected_tokens,
        }
        if visible_new_tokens is not None:
            s["visible_new_tokens"] = visible_new_tokens
        return s
