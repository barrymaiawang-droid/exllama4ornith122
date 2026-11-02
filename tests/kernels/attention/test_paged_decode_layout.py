"""
Dense (query, head) row layout of the paged decode split / combine kernels (triton_paged.decode_row_layout): every
GQA group size, multi-token query length, head_dim and cache width (fp16 and packed 4-bit) must give the same
result as the prefill kernel on the same cache, with the output landing in the right (query, head) slot, in the
single-split (FINAL) and the combine path alike. A wrong row mapping shows up as rows swapped between heads or
query positions, which the prefill kernel (its own row tiling) does not share.
"""
import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import (
    decode_row_layout, paged_attn_triton_decode, paged_attn_triton_prefill,
)
from testlib.attention import rand_packed_cache


def make_caches(n_kv, head_dim, bits, num_pages, page_size, device):
    if bits:
        return rand_packed_cache(num_pages, page_size, n_kv, head_dim, bits, device)
    kc = torch.randn((num_pages, page_size, n_kv, head_dim), dtype = torch.float16, device = device)
    vc = torch.randn((num_pages, page_size, n_kv, head_dim), dtype = torch.float16, device = device)
    return kc, vc, None


@pytest.mark.nogpu
def test_row_layout_shapes():
    # 5-token verify over a 6-head group: one 32-row program
    assert decode_row_layout(5, 6, 128) == (32, 1)
    assert decode_row_layout(5, 6, 256) == (32, 1)
    assert decode_row_layout(1, 6, 128) == (16, 1)
    assert decode_row_layout(1, 16, 128) == (16, 1)
    if torch.version.hip is not None:
        assert decode_row_layout(8, 16, 256, True) == (16, 8)
        assert decode_row_layout(13, 16, 256) == (32, 7)
        assert decode_row_layout(8, 16, 256) == (32, 4)
    assert decode_row_layout(8, 8, 128) == (32, 2)
    assert decode_row_layout(1, 1, 256) == (16, 1)


# Generator-specific row parity is covered by the bucket-boundary test below.

@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("bits", [0, 4])
@pytest.mark.parametrize("q_len", [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize("group", [1, 2, 6, 8, 16])
@pytest.mark.parametrize("num_splits", [1, 16])
@torch.inference_mode()
def test_decode_layout_vs_prefill(device, head_dim, bits, q_len, group, num_splits):
    torch.manual_seed(group * 1000 + q_len * 10 + bits)
    n_kv = 2
    n_q = n_kv * group
    bsz, ctx, page_size = 2, 1024, 256
    num_pages = ctx // page_size
    kc, vc, qc = make_caches(n_kv, head_dim, bits, bsz * num_pages, page_size, device)
    # Distinct per-head, per-position queries so a misplaced row cannot match by accident
    q = torch.randn((bsz, q_len, n_q, head_dim), dtype = torch.float16, device = device)
    bt = torch.arange(bsz * num_pages, dtype = torch.int32, device = device).view(bsz, num_pages)
    sl = torch.tensor([ctx - q_len, ctx - q_len - 37], dtype = torch.int32, device = device)
    common = dict(q = q, k = None, v = None, k_cache = kc, v_cache = vc, block_table = bt, cache_seqlens = sl,
                  causal = True, qc = qc, max_kv_len = ctx, pre_appended_len = q_len,
                  n_kv_heads_override = n_kv if bits else None)
    o_dec = paged_attn_triton_decode(**common, num_splits = num_splits)
    o_pre = paged_attn_triton_prefill(**common)
    torch.testing.assert_close(o_dec, o_pre, rtol = 0, atol = 2e-3)


@pytest.mark.parametrize("q_len,causal", [(5, False), (8, True)])
@pytest.mark.parametrize("bits", [0, 4])
@torch.inference_mode()
def test_group16_decode_sliding_vs_prefill(device, q_len, causal, bits):
    # Group-16 ROCm decode uses a query-major row layout in both causal and
    # noncausal modes. Compare the mathematical result, not bit parity.
    torch.manual_seed(1600 + q_len + bits)
    bsz, n_kv, n_q, head_dim, ctx, page_size = 1, 2, 32, 256, 1024, 256
    pages = ctx // page_size
    kc, vc, qc = make_caches(n_kv, head_dim, bits, pages, page_size, device)
    q = torch.randn((bsz, q_len, n_q, head_dim), dtype = torch.float16, device = device)
    bt = torch.arange(pages, dtype = torch.int32, device = device).view(bsz, pages)
    sl = torch.tensor([ctx - q_len], dtype = torch.int32, device = device)
    common = dict(q = q, k = None, v = None, k_cache = kc, v_cache = vc, block_table = bt, cache_seqlens = sl,
                  causal = causal, window_size = (256, 0 if causal else -1), qc = qc,
                  max_kv_len = ctx, pre_appended_len = q_len, n_kv_heads_override = n_kv if bits else None)
    o_dec = paged_attn_triton_decode(**common, num_splits = 16)
    o_pre = paged_attn_triton_prefill(**common)
    torch.testing.assert_close(o_dec, o_pre, rtol = 0, atol = 2e-3)


@pytest.mark.parametrize("prefix", [4094, 4095, 4096, 24554, 24555, 24568, 24575, 24576])
@pytest.mark.skipif(torch.version.hip is None, reason = "ROCm generator bucket parity")
@torch.inference_mode()
def test_generator_bucket_matches_sequential_decode(device, prefix):
    # Generator tables round their live width to 16 pages; a verifier table
    # includes draft lookahead and may enter the next bucket before AR does.
    torch.manual_seed(11)
    page, bucket, n_kv, n_q, hd, q_len = 256, 16, 2, 32, 256, 8
    pages = (prefix + q_len + page - 1) // page
    kc, vc, _ = make_caches(n_kv, hd, 0, pages, page, device)
    q = torch.randn((1, q_len, n_q, hd), dtype = torch.float16, device = device)

    def table_for(seq_len):
        width = ((seq_len + page * bucket - 1) // (page * bucket)) * bucket
        table = torch.zeros((1, width), dtype = torch.int32, device = device)
        valid = min(width, pages)
        table[0, :valid] = torch.arange(valid, dtype = torch.int32, device = device)
        return table

    block = paged_attn_triton_decode(
        q, None, None, kc, vc, table_for(prefix + 1 + 15 + 7),
        torch.tensor([prefix], dtype = torch.int32, device = device),
        causal = True, pre_appended_len = q_len, _page_bucket = bucket,
    )
    # A configured drafter can temporarily produce no draft tokens, leaving a
    # one-row target call whose table still includes the configured lookahead.
    one_with_drafter = paged_attn_triton_decode(
        q[:, :1].contiguous(), None, None, kc, vc, table_for(prefix + 1 + 15),
        torch.tensor([prefix], dtype = torch.int32, device = device),
        causal = True, pre_appended_len = 1, _page_bucket = bucket,
    )
    for row in range(q_len):
        ar = paged_attn_triton_decode(
            q[:, row:row + 1].contiguous(), None, None, kc, vc, table_for(prefix + row + 1),
            torch.tensor([prefix + row], dtype = torch.int32, device = device),
            causal = True, pre_appended_len = 1,
        )
        assert torch.equal(block[:, row:row + 1], ar), (prefix, row)
        if row == 0:
            assert torch.equal(one_with_drafter, ar), prefix

@pytest.mark.nogpu
def test_scope_layout_unmarked_and_generator_marker(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    assert decode_row_layout(13, 16, 256) == (32, 7)
    assert decode_row_layout(8, 16, 256) == (32, 4)
    assert decode_row_layout(8, 16, 256, True) == (16, 8)
    assert decode_row_layout(1, 16, 256) == decode_row_layout(1, 16, 256, True) == (16, 1)

@pytest.mark.nogpu
def test_quant_fallback_forwards_generator_marker(monkeypatch):
    from types import SimpleNamespace
    from exllamav3.modules.attention_fn import triton_paged
    observed = {}
    monkeypatch.setattr(triton_paged, "paged_attn_triton_decode", lambda **kw: observed.update(kw) or "called")
    args = SimpleNamespace(q_cache=(1, 2, 3, 4, 4, 4), q_len=8, dim=256,
        q=torch.empty((1,), dtype=torch.float16), non_causal_spans=None,
        block_table=None, cache_seqlens=None, causal=True, sm_scale=1.0,
        get_window_size=lambda: (-1, -1), softcap=0, sinks=None,
        num_kv_heads=2, max_kv_len=None, block_table_page_bucket=16)
    assert triton_paged.fn_triton_paged_attn_decode_qc(args) == "called"
    assert observed["_page_bucket"] == 16
