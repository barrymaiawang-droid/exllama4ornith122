"""
Repetition guard on the real engine (Job(repetition_guard = ...)): a scripted sampler makes the model's output
loop, the guard holds, rolls back, nudges and stops, and the gates hold end to end:

- the streamed text is never retracted and equals the final text and the decode of the visible token ids;
- the visible output carries no nudge token;
- with the guard off, the same script produces today's output, byte for byte;
- spec + guard == plain + guard on the internal token stream (n-gram drafting on, the only drafter a dense
  model test can switch on without a second model);
- on a recurrent model the rollback replays or rewinds the state and generation continues.

The sampler honours the logit mask the engine passes (dense -inf or packed bits), which is how a rollback's
one-step ban reaches it. Model roles "dense-llama" and "recurrent".
"""
import pytest
import torch

from exllamav3 import Generator, Job
from exllamav3.generator.sampler import Sampler
from testlib.e2e import load_model

LOOP_TEXT = " The quick brown fox jumps over the lazy dog and then"
TAIL_TEXT = " finally the story ends with a quiet evening by the sea."
OPTS = dict(min_match = 4, confirm = 6, max_interventions = 2, escalate_within = 64)


class ScriptedSampler(Sampler):
    """Returns scripted token ids; a banned token (masked out) falls back to its alternative"""

    def __init__(self, script: list[int], alt: int):
        super().__init__()
        self.script = script
        self.alt = alt
        self.i = 0
        self.banned_hits = 0

    def forward(self, logits, sequence_ids = None, rand_u32 = None, tokenizer = None, logit_mask = None,
                return_state = False, **kwargs):
        t = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if logit_mask is not None:
            m = logit_mask.flatten()
            if m.dtype == torch.int32:
                blocked = (int(m[t >> 5].item()) >> (t & 31)) & 1 == 0
            else:
                blocked = bool(torch.isinf(m[t]).item())
            if blocked:
                self.banned_hits += 1
                t = self.alt
        return torch.full(logits.shape[:-1], t, dtype = torch.long, device = logits.device)


def run(gen, job):
    gen.enqueue(job)
    text, events = "", []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            text += r.get("text", "")
            events.append(r)
    return text, events


def visible_ids(events):
    out = []
    for r in events:
        if "token_ids" in r:
            out += r["token_ids"].flatten().tolist()
    return out


def guard_events(events):
    return [r["repetition_guard_event"] for r in events
            if "repetition_guard_event" in r and "tier" in r["repetition_guard_event"]]


def make_script(tok, loops: int):
    loop = tok.encode(LOOP_TEXT, add_bos = False)[0].tolist()
    tail = tok.encode(TAIL_TEXT, add_bos = False)[0].tolist()
    alt = tok.single_id(" however")
    # the model "loops": the phrase, then copies of it; after each rollback the script resumes copying
    return loop * (loops + 1) + tail, alt, loop, tail


@pytest.fixture(scope = "module")
def lm(model_registry, device):
    with load_model(model_registry.get("dense-llama").path, device, cache_tokens = 4096) as lm:
        yield lm


@pytest.fixture
def gen(lm):
    g = Generator(model = lm.model, cache = lm.cache, tokenizer = lm.tokenizer, max_batch_size = 1)
    yield g
    g.close()


@pytest.mark.model("dense-llama")
@torch.inference_mode()
def test_guard_off_is_unchanged_and_tiers_run_on_the_engine(gen, lm):
    tok = lm.tokenizer
    ids = tok.encode("Once upon a time,", add_bos = True)
    script, alt, loop, tail = make_script(tok, loops = 3)

    # baseline: the loop streams through untouched
    plain, plain_events = run(gen, Job(input_ids = ids, max_new_tokens = len(script), sampler = ScriptedSampler(script, alt)))
    assert plain == tok.decode(torch.tensor([script]))[0]
    assert plain_events[-1]["eos_reason"] == "max_new_tokens"

    # guarded: hold, tier 1, tier 2 (hidden nudge), tier 3
    sampler = ScriptedSampler(script, alt)
    job = Job(input_ids = ids, max_new_tokens = len(script) + 64, sampler = sampler, repetition_guard = OPTS)
    text, events = run(gen, job)
    final = events[-1]
    assert final["eos"] and final["eos_reason"] == "repetition"
    tiers = [e["tier"] for e in guard_events(events)]
    assert tiers == [1, 2, 3], tiers
    assert sampler.banned_hits >= 2, "the one-step ban after each rollback reached the sampler"

    # gates
    assert text == final["full_completion"], "streamed text equals the final text (nothing retracted)"
    vis = visible_ids(events)
    assert tok.decode(torch.tensor([vis]))[0] == text, "visible token ids decode to the visible text"
    nudge = job.repetition_guard.nudge_ids(tok)[0].tolist()
    seq = job.sequences[0].sequence_ids.torch()[0, ids.shape[-1]:].tolist()
    assert any(seq[i:i + len(nudge)] == nudge for i in range(len(seq))), "the nudge is in the model's context"
    assert not any(vis[i:i + len(nudge)] == nudge for i in range(len(vis))), "the nudge is not in the visible ids"
    assert job.repetition_guard.nudge_text.strip() not in text
    s = final["repetition_guard"]
    assert s["tier"] == 3 and s["injected_tokens"] == len(nudge)
    assert s["visible_new_tokens"] == len(vis) == final["new_tokens"] - len(nudge)
    # the loop phrase appears far fewer times than in the unguarded run
    assert text.count(LOOP_TEXT) < plain.count(LOOP_TEXT)


@pytest.mark.model("dense-llama")
@torch.inference_mode()
def test_spec_plus_guard_equals_plain_plus_guard(lm):
    """n-gram drafting on and off: the same visible text, the same internal token stream, the same tiers"""
    tok = lm.tokenizer
    ids = tok.encode("Once upon a time,", add_bos = True)
    script, alt, _, _ = make_script(tok, loops = 3)
    outs = []
    for ngram in (0, 3):
        g = Generator(model = lm.model, cache = lm.cache, tokenizer = lm.tokenizer, max_batch_size = 1,
                      ngram_match_min = ngram)
        try:
            job = Job(input_ids = ids, max_new_tokens = len(script) + 64, sampler = ScriptedSampler(script, alt),
                      repetition_guard = OPTS)
            text, events = run(g, job)
            outs.append((text, job.sequences[0].sequence_ids.torch()[0].tolist(),
                         [e["tier"] for e in guard_events(events)], events[-1]["eos_reason"]))
        finally:
            g.close()
    assert outs[0] == outs[1]


@pytest.mark.model("recurrent")
@torch.inference_mode()
def test_rollback_on_a_recurrent_model(model_registry, device):
    """The rollback rewinds the recurrent state in place or restores a stash and replays; generation continues"""
    with load_model(model_registry.get("recurrent").path, device, cache_tokens = 4096) as lm:
        tok = lm.tokenizer
        g = Generator(model = lm.model, cache = lm.cache, tokenizer = lm.tokenizer, max_batch_size = 1)
        try:
            ids = tok.encode("Once upon a time,", add_bos = True)
            script, alt, loop, tail = make_script(tok, loops = 1)
            sampler = ScriptedSampler(script, alt)
            job = Job(input_ids = ids, max_new_tokens = len(script) + 16, sampler = sampler,
                      repetition_guard = dict(OPTS, max_interventions = 1, nudge = None))
            text, events = run(g, job)
            tiers = [e["tier"] for e in guard_events(events)]
            assert tiers and tiers[0] == 1
            assert text == events[-1]["full_completion"]
            assert tok.decode(torch.tensor([visible_ids(events)]))[0] == text
            assert tail[-1] == job.sequences[0].sequence_ids.torch()[0, -1].item() or events[-1]["eos_reason"] == "repetition"
        finally:
            g.close()
