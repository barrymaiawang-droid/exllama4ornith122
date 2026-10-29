"""
Repetition guard: detect a job copying its own output and steer or stop it, invisibly to the client.

Tiers (Job(repetition_guard = {...}), off unless asked for):

    0  detect   the last `min_match` tokens equal an earlier span of this turn's own output. Output is
                held back from here (like a partial banned-string match) while the copy keeps extending;
                it is released untouched the moment the copy breaks
    1  escape   the copy reached `confirm` tokens (or one full period): roll the job back to the first
                held token and ban that token for one step
    2  nudge    a loop again within `escalate_within` tokens of tier 1: roll back again and inject the
                model-voiced interjection `nudge`, tagged hidden, so it steers the model but never reaches
                the client's text, token ids or completion
    3  stop     a loop after the nudge, or the `max_interventions` budget spent: end the job with
                eos_reason "repetition"

Nothing streamed is ever retracted: holding starts at the token whose arrival completed the first
`min_match`-token match, so the client has seen at most `min_match - 1` tokens of a copy, and a rollback
discards held tokens only. The detector matches token ids against the turn's own output only, so a model
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
    "min_match": 8,              # tokens that must match an earlier span before output is held (tier 0)
    "confirm": 32,               # a copy this long, or one full period if shorter, is a loop (tier 1)
    "max_interventions": 2,      # rollbacks (tier 1 / 2) per request before tier 3
    "escalate_within": 64,       # a loop this soon after an intervention escalates to the next tier
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

    def __init__(self, min_match: int = 8, confirm: int = 32):
        assert min_match >= 2, "min_match must be at least 2"
        assert confirm >= 1, "confirm must be at least 1"
        self.min_match = min_match
        self.confirm = confirm
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
        """Copy length that counts as a loop: one full period, capped at `confirm`, never below the trigger"""
        return max(self.min_match, min(period, self.confirm))

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
                kind = "confirm" if self.matched >= self.confirm_length(self.period) else "extend"
                return GuardEvent(kind, self.period, self.matched, self.hold_start,
                                  self.hold_start - self.min_match + 1, self.hold_start - self.min_match + 1 - self.period)
            self._reset()
            released = True

        g = self._gram(i)
        earlier = self._index.get(g) if g is not None else None
        last_end = earlier[-1] if earlier else None
        self._register(i)
        if last_end is None:
            return GuardEvent("release" if released else "none", released = released)

        # The most recent earlier occurrence gives the period; the copy began min_match tokens ago
        period = i - last_end
        self.period = period
        self.matched = self.min_match
        self.hold_start = i
        copy_start = i - self.min_match + 1
        kind = "confirm" if self.matched >= self.confirm_length(period) else "suspect"
        return GuardEvent(kind, period, self.matched, i, copy_start, copy_start - period, released = released)

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
        self.detector = SuffixLoopDetector(opts["min_match"], opts["confirm"])
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
