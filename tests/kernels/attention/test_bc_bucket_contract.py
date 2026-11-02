"""Generator table-bucket eligibility through the actual BC attention step."""
from types import SimpleNamespace

import pytest
import torch

from exllamav3.modules.attention_fn.bc_attn import BCAttn

pytestmark = pytest.mark.nogpu


def make_bc(window, calls, heads = 32, kv_heads = 2):
    bca = BCAttn.__new__(BCAttn)
    bca.qsa = False
    bca.slot_widths = {}
    bca.window_size = window
    bca.num_q_heads, bca.num_kv_heads = heads, kv_heads
    bca.hidden_size, bca.o_dtype = 3072, torch.float16
    bca.bc = SimpleNamespace(run = lambda *args: None)
    bca._configure = lambda bsz, q_len, causal, regime, page_bucket: calls.append(page_bucket)
    return bca


def step(bca, q_len, marker, *, bsz = 1, causal = True):
    x = torch.zeros((bsz, q_len, 3072), dtype = torch.float16)
    block = torch.zeros((bsz, 16), dtype = torch.int32)
    lengths = torch.zeros((bsz,), dtype = torch.int32)
    return bca.step(x, lengths, block, 0, None, None, None, causal = causal, page_bucket = marker)


@pytest.mark.parametrize("window,expected", [(None, 16), (-1, 16), ((-1, -1), 16), (256, 0), ((256, 0), 0)])
@pytest.mark.parametrize("q_len", [1, 8])
def test_generator_bucket_window_and_slot_reconfiguration(monkeypatch, window, expected, q_len):
    monkeypatch.setattr(torch.version, "hip", "test")
    calls = []
    bca = make_bc(window, calls)
    step(bca, q_len, 16)
    assert calls == [expected]
    step(bca, q_len, 16)
    assert calls == [expected]
    step(bca, q_len, 0)
    assert calls == ([16, 0] if expected else [0])


@pytest.mark.parametrize("bsz,causal,heads,marker", [(2, True, 32, 16), (1, False, 32, 16),
                                                       (1, True, 16, 16), (1, True, 32, 0),
                                                       (1, True, 32, 8)])
def test_generator_bucket_requires_supported_scope(monkeypatch, bsz, causal, heads, marker):
    monkeypatch.setattr(torch.version, "hip", "test")
    calls = []
    bca = make_bc(-1, calls, heads = heads)
    step(bca, 8, marker, bsz = bsz, causal = causal)
    assert calls == [0]


def test_generator_bucket_declines_qsa_and_non_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    calls = []
    bca = make_bc(-1, calls)
    bca.qsa, bca.qsa_threshold = True, 100
    x = torch.zeros((1, 8, 3072), dtype = torch.float16)
    block = torch.zeros((1, 16), dtype = torch.int32)
    lengths = torch.zeros((1,), dtype = torch.int32)
    bca.step(x, lengths, block, 0, None, None, None, causal = True,
             host_seqlens = lengths, page_bucket = 16)
    assert calls == [0]

    monkeypatch.setattr(torch.version, "hip", None)
    calls = []
    bca = make_bc(-1, calls)
    step(bca, 8, 16)
    assert calls == [0]
