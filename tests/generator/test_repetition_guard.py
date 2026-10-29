"""
Repetition guard (generator/repetition_guard.py, Job(repetition_guard = ...)) without a model: the detector and
tier logic on token lists, and the Job's sampling path driven directly with a stub generator and tokenizer. The
scripted "model" honours the one-step ban a rollback leaves behind (what prepare_logit_mask would mask) and the
hidden injection the way receive_logits does, so the whole guard loop runs: hold, release, rollback, nudge, stop.

Gates: the streamed text is never retracted and equals the final text; the visible output contains no nudge
tokens; with the guard off (or in shadow mode) the output is byte-identical to the plain stream.
"""
from types import SimpleNamespace

import pytest
import torch

from exllamav3.generator.job import Job
from exllamav3.generator.repetition_guard import SuffixLoopDetector, RepetitionGuard, DEFAULTS
from exllamav3.util.tensor import SeqTensor

pytestmark = pytest.mark.nogpu

P = [10, 11, 12, 13, 14, 15, 16, 17]        # the phrase the model loops on
NUDGE_IDS = [900, 901, 902]
# Job tests: a copy of 6 tokens is a loop whatever its period, so short scripts confirm
OPTS = dict(min_match = 4, confirm = 6, max_interventions = 2, escalate_within = 64)


# ---------------------------------------------------------------- detector and tiers

def kinds(det, tokens):
    return [det.feed(t).kind for t in tokens]


def test_detector_suspects_holds_and_confirms_one_period():
    d = SuffixLoopDetector(min_match = 4, confirm = 32)
    k = kinds(d, P + P)
    # the first copy: three tokens pass unnoticed, the fourth completes a 4-gram seen before (suspect), the copy
    # extends, and one full period confirms
    assert k[:8] == ["none"] * 8
    assert k[8:] == ["none", "none", "none", "suspect", "extend", "extend", "extend", "confirm"]
    ev = d.feed(P[0])
    assert ev.kind == "confirm" and ev.period == 8 and ev.hold_start == 11 and ev.copy_start == 8 and ev.source_start == 0


def test_detector_short_period_confirms_at_the_trigger():
    d = SuffixLoopDetector(min_match = 4, confirm = 32)
    assert kinds(d, [9, 7, 7, 7, 7, 7]) == ["none"] * 5 + ["confirm"]
    assert d.confirm_length(1) == 4 and d.confirm_length(8) == 8 and d.confirm_length(100) == 32


def test_detector_long_period_is_capped_by_confirm():
    d = SuffixLoopDetector(min_match = 8, confirm = 32)
    k = kinds(d, list(range(100, 140)) * 2)
    assert k.index("suspect") == 47                    # 8th token of the copy
    assert k.index("confirm") == 71                    # 32 matched tokens, not the 40-token period
    assert all(x == "extend" for x in k[48:71])


def test_detector_releases_on_a_break_and_truncates_cleanly():
    d = SuffixLoopDetector(min_match = 4, confirm = 32)
    assert kinds(d, P + P[:6] + [99]) == ["none"] * 11 + ["suspect", "extend", "extend", "release"]
    assert not d.suspecting
    # a break that is itself the start of another copy: one event, flagged released, so the old hold is let go
    d1 = SuffixLoopDetector(min_match = 3, confirm = 32)
    kinds(d1, [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5])
    ev = d1.feed(1)                                     # breaks the copy of 1..5 and starts copying 5,1,2? no: 4,5,1 is new
    assert ev.kind == "release" and ev.released
    d2 = SuffixLoopDetector(min_match = 3, confirm = 32)
    k = kinds(d2, [1, 2, 3, 4, 5, 1, 2, 3, 9, 7, 1, 2, 3])
    assert k[-1] == "suspect"                           # copying "1 2 3 9" (the most recent earlier "1 2 3")
    ev = d2.feed(4)                                     # that copy breaks, and "2 3 4" was seen at the start
    assert ev.kind == "suspect" and ev.released and ev.period == 10
    d2 = SuffixLoopDetector(min_match = 4, confirm = 32)
    kinds(d2, P + P[:5])
    assert d2.suspecting
    d2.truncate(11)                                     # a rollback to the first held token
    assert len(d2) == 11 and not d2.suspecting
    assert kinds(d2, [50, 51, 52, 53]) == ["none"] * 4
    # the forgotten grams are gone from the index: re-feeding the copy re-suspects at the same place
    d3 = SuffixLoopDetector(min_match = 4, confirm = 32)
    kinds(d3, P + P[:5]); d3.truncate(8)
    assert kinds(d3, P[:4]) == ["none", "none", "none", "suspect"]


def test_detector_ignores_copies_of_what_it_never_saw():
    d = SuffixLoopDetector(min_match = 4, confirm = 32)
    assert kinds(d, P) == ["none"] * 8                  # the prompt is not fed, so quoting it is no copy


def test_guard_tiers_budget_and_escalation():
    g = RepetitionGuard(dict(OPTS, confirm = 32))
    ev = SuffixLoopDetector(4, 32).feed(0)              # any event will do for record()
    for _ in range(10): g.feed(1)                       # position 10
    assert g.plan(allow_nudge = True) == 1
    g.record(1, ev)
    for _ in range(20): g.feed(1)
    assert g.plan(allow_nudge = True) == 2              # again within escalate_within -> nudge
    assert g.plan(allow_nudge = False) == 3             # ... unless inside a constrained span
    g.record(2, ev)
    assert g.plan(allow_nudge = True) == 3              # budget of 2 spent
    far = RepetitionGuard(dict(OPTS, confirm = 32))
    far.record(1, ev)
    for _ in range(200): far.feed(1)
    assert far.plan(allow_nudge = True) == 1            # a loop long after the last intervention starts over
    assert RepetitionGuard(dict(OPTS, nudge = None)).plan(allow_nudge = True) == 1
    with pytest.raises(AssertionError):
        RepetitionGuard({"bogus": 1})
    assert RepetitionGuard(True).options == DEFAULTS


def test_guard_shadow_mode_notes_once_per_confirm_window():
    g = RepetitionGuard(dict(OPTS, mode = "shadow", confirm = 32))
    confirms = [g.feed(t).kind for t in P * 10].count("confirm")
    assert 1 <= confirms <= 3, confirms              # one per confirm window, not one per token


# ---------------------------------------------------------------- Job on a stub generator

class Tok:
    def get_id_to_piece_list(self, special = False):
        return [f" {i}" for i in range(1024)]
    def decode(self, ids, decode_special_tokens = False):
        return ["".join(f" {t}" for t in ids.flatten().tolist())]
    def encode(self, text, add_bos = False, encode_special_tokens = False):
        return torch.tensor([NUDGE_IDS])


class Page:
    def __init__(self, kv_position):
        self.kv_position = kv_position
        self.can_revert = True
        self.phash = b"p"
        self.prev_hash = None
    def update_hash(self, h):
        self.phash = h


def make_job(**kwargs):
    prompt = torch.tensor([[1, 2, 3]])
    job = Job(input_ids = prompt, max_new_tokens = 1000, **kwargs)
    job.generator = SimpleNamespace(tokenizer = Tok(), recurrent_cache = None, draft_model = None, ngram_match_min = 0,
                                    hybrid_draft_tokens = 0, padded_vocab_size = 1024, mtp_draft = False)
    job.max_rq_tokens, job.rq_margin, job.rq_headroom = 10 ** 9, 0, 0
    job.held_text = ""
    job.held_tokens = SeqTensor((1, 0), dtype = torch.long, seq_dim = -1)
    job.held_k_tokens = SeqTensor((1, 0, 0), dtype = torch.long, seq_dim = 1)
    job.held_k_probs = SeqTensor((1, 0, 0), dtype = torch.float, seq_dim = 1)
    job.held_probs = SeqTensor((1, 0), dtype = torch.float, seq_dim = -1)
    job.held_logits = SeqTensor((1, 0, 1024), dtype = torch.float, seq_dim = 1)
    job.full_completion = ""
    job.sam = None
    job.time_enqueue = job.time_first_prefill = job.time_first_token = 0.0
    seq = job.sequences[0]
    seq.kv_position = prompt.shape[-1]
    seq.allocated_pages = [Page(prompt.shape[-1])]
    return job


def run(job, script):
    """
    Drive receive_sample with a scripted model. script: (token, alternative) pairs; the alternative is sampled
    when the token is banned by a rollback's checkpoint (prepare_logit_mask would mask it). Pending forced tokens
    (the hidden nudge) are consumed first, as receive_logits does. Returns the streamed text and all events.
    """
    text, events, i = "", [], 0
    while True:
        if job.forced_ids is not None:
            tok = job._pop_forced_token("cpu")
        else:
            if i >= len(script):
                break
            t, alt = script[i]
            i += 1
            if job.checkpoint and job.checkpoint["offset"] == 0 and t in job.checkpoint["explored_tokens"]:
                t = alt
            tok = torch.tensor([[t]])
        results = []
        eos, _, _ = job.receive_sample(None, tok, None, None, None, results)
        for r in results:
            text += r.get("text", "")
        events += results
        if eos:
            break
    return text, events


def pieces(tokens):
    return "".join(f" {t}" for t in tokens)


def visible_ids(events):
    out = []
    for r in events:
        if "token_ids" in r:
            out += r["token_ids"].flatten().tolist()
    return out


def script(tokens, alt = 50):
    return [(t, alt) for t in tokens]


def test_guard_off_streams_the_loop_unchanged():
    job = make_job()
    text, events = run(job, script(P * 3))
    assert text == pieces(P * 3)
    assert job.checkpoint is None and job.repetition_guard is None
    assert not any("repetition_guard_event" in r for r in events)


def test_tier1_rolls_back_the_held_copy_and_bans_its_first_token():
    job = make_job(repetition_guard = OPTS)
    text, events = run(job, script(P + P[:6] + [13] + [51, 52, 53, 54], alt = 50))
    # P streams; the copy's first three tokens were out before the match completed; the three held copy tokens
    # (13, 14, 15; the sixth copied token confirms) are discarded by the rollback; 13 is banned for one step so
    # the alternative 50 is sampled
    assert text == pieces(P + P[:3] + [50, 51, 52, 53, 54])
    assert text == job.full_completion
    assert job.sequences[0].sequence_ids.torch()[0, 3:].tolist() == P + P[:3] + [50, 51, 52, 53, 54]
    rb = [r for r in events if "repetition_guard_event" in r]
    assert len(rb) == 1
    e = rb[0]["repetition_guard_event"]
    assert e["tier"] == 1 and e["period"] == 8 and e["matched"] == 6 and e["rollback_tokens"] == 3
    assert rb[0]["suppressed_tokens"].flatten().tolist() == P[3:6]
    assert job.checkpoint is None, "the one-step ban is released once a non-copy token is sampled"
    assert visible_ids(events) == P + P[:3] + [50, 51, 52, 53, 54]


def test_text_is_held_during_suspicion_and_released_on_a_break():
    job = make_job(repetition_guard = OPTS)
    seen = []
    text = ""
    for t in P + P[:5] + [99, 60, 61]:
        results = []
        job.receive_sample(None, torch.tensor([[t]]), None, None, None, results)
        step_text = "".join(r.get("text", "") for r in results)
        text += step_text
        seen.append(bool(step_text))
    # tokens 11 and 12 of the output (the 4th and 5th copied) are held, 99 breaks the copy and releases them at once
    assert seen[:11] == [True] * 11 and seen[11:13] == [False] * 2 and seen[13]
    assert text == pieces(P + P[:5] + [99, 60, 61]) == job.full_completion
    assert job.checkpoint is None and job.repetition_guard.interventions == []


def test_tier2_injects_a_hidden_nudge_that_never_reaches_the_client():
    job = make_job(repetition_guard = OPTS)
    seq_script = P + P[:6] + [13] + P[:6] + [13] + [60, 61, 62]
    text, events = run(job, script(seq_script, alt = 50))
    inter = [r["repetition_guard_event"] for r in events if "repetition_guard_event" in r and "tier" in r["repetition_guard_event"]]
    assert [e["tier"] for e in inter] == [1, 2]
    assert inter[1]["injected_tokens"] == 3 and inter[1]["rollback_tokens"] == 3
    hidden = [r["repetition_guard_event"]["hidden_token_id"] for r in events
              if "repetition_guard_event" in r and "hidden_token_id" in r["repetition_guard_event"]]
    assert hidden == NUDGE_IDS
    for n in NUDGE_IDS:
        assert f" {n}" not in text and n not in visible_ids(events)
    assert text == job.full_completion
    ids = job.sequences[0].sequence_ids.torch()[0, 3:].tolist()
    assert NUDGE_IDS == ids[ids.index(900):ids.index(900) + 3], "the nudge is in the model's context"
    # after the nudge the ban on 13 still applied to the first sampled token, then normal sampling resumed
    assert ids[ids.index(902) + 1:] == [50, 60, 61, 62]
    assert job.forced_ids is None and job.forced_hidden_n == 0
    g = job.repetition_guard
    assert g.injected_tokens == 3 and g.rollback_tokens == 6 and len(g.interventions) == 2


def test_tier3_stops_with_eos_reason_repetition_and_a_summary():
    job = make_job(repetition_guard = OPTS)
    text, events = run(job, script(P + P[:6] + [13] + P[:6] + [13] + P[:6] + [70, 71], alt = 50))
    final = events[-1]
    assert final["eos"] and final["eos_reason"] == "repetition"
    s = final["repetition_guard"]
    assert s["tier"] == 3 and [e["tier"] for e in s["interventions"]] == [1, 2, 3]
    assert s["injected_tokens"] == 3 and s["rollback_tokens"] == 9
    assert s["visible_new_tokens"] == final["new_tokens"] - 3
    assert text == final["full_completion"]
    assert text.endswith(pieces([50] + P[:3])), "the confirmed copy was discarded before stopping, its visible head stays"
    assert len(visible_ids(events)) == s["visible_new_tokens"]


def test_tier2_is_skipped_inside_a_constrained_span():
    job = make_job(repetition_guard = OPTS)
    job.filters = [SimpleNamespace(is_active = False, trigger_token = 5, feed = lambda t: False, rewind = lambda n: None)]
    text, events = run(job, script(P + P[:6] + [13] + P[:6] + [13] + [70], alt = 50))
    inter = [r["repetition_guard_event"]["tier"] for r in events
             if "repetition_guard_event" in r and "tier" in r["repetition_guard_event"]]
    assert inter == [1, 3], "no text is injected into a filtered (tool / JSON) span: escape, then stop"
    assert events[-1]["eos_reason"] == "repetition"
    assert all(f" {n}" not in text for n in NUDGE_IDS)


def test_shadow_mode_only_observes():
    plain, _ = run(make_job(), script(P * 4))
    job = make_job(repetition_guard = dict(OPTS, mode = "shadow"))
    text, events = run(job, script(P * 4))
    assert text == plain == job.full_completion
    assert job.checkpoint is None
    g = job.repetition_guard
    assert g.detections >= 1 and g.interventions == [] and g.rollback_tokens == 0
    assert not any("repetition_guard_event" in r for r in events)
    assert events[-1].get("eos") is not True


def test_stop_token_inside_a_hold_ends_the_job_with_the_held_text():
    job = make_job(repetition_guard = OPTS, stop_conditions = [777])
    text, events = run(job, script(P + P[:5] + [777]))
    assert events[-1]["eos"] and events[-1]["eos_reason"] == "stop_token"
    # as for any hold, the text a stop token interrupts is handed back under "held", not streamed
    assert text == pieces(P + P[:3]) == job.full_completion
    assert events[-1]["held"]["text"] == pieces(P[3:5] + [777])
