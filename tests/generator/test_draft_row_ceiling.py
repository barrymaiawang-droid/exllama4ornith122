"""
Verification-row ceiling (Generator.draft_max_rows / EXL3_DRAFT_MAX_ROWS).

The target's speculative verification pass runs bsz * (1 + window) rows. On ROCm, passes of up to 8
rows stay on the kernel tiers whose per-row arithmetic is the single-row decode's; wider passes
switch tiers and their scores drift from AR decoding (docs/ornith122-dflash-parity-handoff-20261102.md).
These tests pin the arithmetic of the ceiling, not the kernels: with a ceiling of 8, no round may
ever verify more than 8 rows, whatever the draft depth or batch, and the EXL3_DRAFT_MAX_ROWS
deployment override must reach the constructor's value.

No GPU or model is required: Generator is instantiated via __new__ with stub jobs.
"""

from types import SimpleNamespace
import os
import pytest
from exllamav3.generator.generator import Generator

pytestmark = pytest.mark.nogpu


class StubJob:
    def __init__(self, sequences = 1, generating = True):
        self.sequences = [None] * sequences
        self.generating = generating

    def is_prefill_done(self):
        return self.generating


def make_generator(jobs, num_draft_tokens, draft_max_rows):
    g = Generator.__new__(Generator)
    g.active_jobs = list(jobs)
    g.num_draft_tokens = num_draft_tokens
    g.draft_max_rows = draft_max_rows
    g._draft_window_last = None
    return g


def verify_rows(g, window):
    """Rows the target pass takes for this window: one input token per generating sequence row plus the window"""
    rows = sum(len(j.sequences) for j in g.active_jobs if j.is_prefill_done())
    return rows * (1 + window)


@pytest.mark.parametrize("depth", [1, 2, 7, 8, 15, 16])
@pytest.mark.parametrize("streams", [1, 2, 3, 4, 8, 9])
def test_ceiling_bounds_every_round(depth, streams):
    g = make_generator([StubJob() for _ in range(streams)], depth, draft_max_rows = 8)
    w = g.row_limited(depth)
    assert 0 <= w <= depth
    # The input tokens alone may exceed the ceiling (nine streams): then nothing is drafted
    assert verify_rows(g, w) <= 8 or w == 0
    # The ceiling is tight: one more draft token per stream would exceed it (unless the depth already fits)
    if w < depth:
        assert verify_rows(g, w + 1) > 8


def test_one_stream_depth_15_verifies_eight_rows():
    """The Node B DFlash2 profile: block-16 drafter (15 proposals), one stream. 16 rows without the ceiling,
    8 with it (7 proposals plus the input token)"""
    g = make_generator([StubJob()], 15, draft_max_rows = 0)
    assert g.row_limited(15) == 15 and verify_rows(g, 15) == 16
    g = make_generator([StubJob()], 15, draft_max_rows = 8)
    assert g.row_limited(15) == 7 and verify_rows(g, 7) == 8


def test_ceiling_off_and_no_generating_rows_keep_the_window():
    g = make_generator([StubJob()], 15, draft_max_rows = 0)
    assert g.row_limited(15) == 15
    g = make_generator([StubJob(generating = False)], 15, draft_max_rows = 8)
    assert g.row_limited(15) == 15      # nothing is verifying this round; prefill-only
    g = make_generator([StubJob()], 15, draft_max_rows = 8)
    assert g.row_limited(0) == 0


def test_multi_sequence_jobs_count_every_row():
    g = make_generator([StubJob(sequences = 4)], 15, draft_max_rows = 8)
    assert g.row_limited(15) == 1 and verify_rows(g, 1) == 8
    g = make_generator([StubJob(sequences = 8)], 15, draft_max_rows = 8)
    assert g.row_limited(15) == 0       # the input tokens alone fill the ceiling: no drafting


def _construct_with_env(monkeypatch, env, arg):
    """Run only the constructor statements that resolve draft_max_rows, with the module's own _env_int"""
    from exllamav3.generator import generator as gen_mod
    for k in ("EXL3_DRAFT_MAX_ROWS",):
        monkeypatch.delenv(k, raising = False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    draft_max_rows = arg
    env_rows = gen_mod._env_int("EXL3_DRAFT_MAX_ROWS")
    if env_rows is not None:
        draft_max_rows = env_rows
    return max(0, draft_max_rows or 0)


def test_env_override_resolution(monkeypatch):
    assert _construct_with_env(monkeypatch, {}, 0) == 0
    assert _construct_with_env(monkeypatch, {}, 32) == 32
    assert _construct_with_env(monkeypatch, {"EXL3_DRAFT_MAX_ROWS": "8"}, 0) == 8
    assert _construct_with_env(monkeypatch, {"EXL3_DRAFT_MAX_ROWS": "8"}, 32) == 8
    assert _construct_with_env(monkeypatch, {"EXL3_DRAFT_MAX_ROWS": "0"}, 32) == 0
    assert _construct_with_env(monkeypatch, {"EXL3_DRAFT_MAX_ROWS": ""}, 32) == 32


def test_constructor_reads_env_override():
    """The constructor source carries the override next to the argument it replaces"""
    import inspect
    src = inspect.getsource(Generator.__init__)
    assert '_env_int("EXL3_DRAFT_MAX_ROWS")' in src
    i = src.index('_env_int("EXL3_DRAFT_MAX_ROWS")')
    j = src.index("self.draft_max_rows = max(0, draft_max_rows or 0)")
    assert i < j, "the override must be applied before draft_max_rows is stored"
