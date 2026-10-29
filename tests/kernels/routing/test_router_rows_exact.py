"""
EXL3_ROUTER_DET=0 (displaced int8 gate, an empty tensor) on ROCm: multi-row router calls of 2..16 rows must
score bit-identically to routing each row on its own, i.e. a speculative verify window selects exactly the
experts AR decoding would. Rows 9..16 run as two 8-row launches of the RDNA router GEMV (routing_gemv_rdna.cuh);
before this, rows 4..16 fell to the rocBLAS hgemm. Larger batches are not covered by the guarantee.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(torch.version.hip is None, reason = "RDNA router GEMV path is ROCm-only")


def route(x, gate, gate_t, gate_i8, K):
    R, E = x.shape[0], gate.shape[1]
    scores = torch.empty((R, E), dtype = torch.half, device = x.device)
    sel = torch.empty((R, K), dtype = torch.long, device = x.device)
    w = torch.empty((R, K), dtype = torch.half, device = x.device)
    ext.routing_std(x, gate, scores, sel, w, None, gate_t, None, gate_i8, None)
    return scores, sel, w


@pytest.mark.parametrize("K_dim, E, K", [(3072, 256, 8), (2048, 128, 8), (4096, 512, 10)])
@torch.inference_mode()
def test_multirow_equals_single_row(device, K_dim, E, K):
    torch.manual_seed(11)
    gate = (torch.randn(K_dim, E, device = device) * (1.0 / K_dim ** 0.5)).half()
    gate_t = gate.T.contiguous()
    displaced = torch.empty(0, dtype = torch.int8, device = device)
    x_all = torch.randn(16, K_dim, device = device).half()
    singles = [route(x_all[r:r+1].contiguous(), gate, gate_t, displaced, K) for r in range(16)]
    for R in range(2, 17):
        s, sel, w = route(x_all[:R].contiguous(), gate, gate_t, displaced, K)
        for r in range(R):
            s1, sel1, w1 = singles[r]
            assert torch.equal(s[r], s1[0]), (R, r)
            assert torch.equal(sel[r], sel1[0]), (R, r)
            assert torch.equal(w[r], w1[0]), (R, r)
