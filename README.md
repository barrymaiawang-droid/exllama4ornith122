# exllama4ornith122

An ExLlamaV3 fork specialised for running **Ornith-1.5-122B EXL3 (4 bpw)** on a single
**AMD gfx1151** (Ryzen AI MAX+ 395 / Radeon 8060S, Strix Halo, unified LPDDR5X) under
**ROCm 10**.

This README is the fork's own. It documents what the fork changes, how the deployment
runs, and — most importantly — **what the hardware can and cannot do**, with measured
numbers. The upstream ExLlamaV3 README has been removed; for general installation,
architecture support, conversion and EXL3 quantization documentation see
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3). Upstream's README
remains in this repository's git history if you need it.

> **Correction to an earlier version of this file.** It claimed **91.6–97.8 t/s** decode
> with DFlash2. That does **not reproduce**. Measured on the deployed build, decode is
> **17–21 t/s** depending on draft depth. See [Measured performance](#measured-performance).
> The error was in the measurement, not the code.

---

## Provenance

| | |
|---|---|
| Upstream | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |
| Base | **v1.6.4**, commit `c28e026b` |
| This branch | `master` = base + **14 commits, 21 files, ~1900 insertions, 43 deletions** |
| Target | gfx1151 only. torch 2.14.0+rocm10.1.0a20260822, HIP 7.16.26332, ROCm 10 |
| License | unchanged from upstream — see [LICENSE](LICENSE) |

ROCm 10 needed **no source patch**. It is the target the fork's changes were built and
measured against; the deltas below are the kernel, router and diagnostic changes.

---

## What this fork changes versus upstream

| Commit | Area | Change |
|---|---|---|
| `434117c1` | ROCm kernel | `gdn.cu`: preserve GDN recurrent-state rounding across speculative tokens (single-row and batched verify paths) |
| `a245af42` | ROCm kernel | 16-row GEMV groups: rows 9–16 split into two independent ≤8-row groups so row arithmetic matches the narrow path; new rowgroup tests |
| `3c885555` | ROCm kernel | same grouping extended to the **multi-matrix** GEMV, opt-in via `EXL3_MGEMV_MAX_M` (default 8; 16 enables it) |
| `6d52fefb` | diagnostics | dispatch route trace for every `exl3_gemm` / `exl3_mgemm` call — C++ ring recorder, Python API, env switches, host tests |
| `dccad676` | diagnostics | include ATen ops in the route-trace accessors |
| `8973706c` | diagnostics | trace the routed-expert tiers (fused / RDNA decode / cooperative); add `util/exact_linear.py` for operator replay |
| `4684eb65` | router | `EXL3_ROUTER_DET` switch: displace the int8 deterministic multi-row router projection so rows >3 route on the half-precision GEMM |
| `999856a6` | docs | handoff note for the remaining DFlash2 numerical failures |
| `13324992` | docs | record DFlash2 as the deployment default with the required `EXL3_ROUTER_DET=0` |
| `74f82137`, `9b0b0706`, `bae03424`, `d7a62d07`, `a5e4acd4` | docs | README sections, change inventory, run and fleet-switch notes |

### Why the router switch exists

Above three rows the router uses an int8 deterministic projection (it exists so replicated
routing agrees across tensor-parallel ranks). On this checkpoint that projection
approximates differently than one-token decoding. On Ornith-1.5-122B EXL3 (48 MoE layers,
`router_type std`, single gfx1151) a 16-row DFlash2 window diverged from AR greedy
decoding: a greedy-equivalence gate at tolerance 0.25 failed at 0.6875 (token 18) and
0.265625 (token 43). With `EXL3_ROUTER_DET=0` the same gate passes — first divergence
moves to token 27 at a 0.03125 near-tie.

**Cost and limits.** A resolution switch, not a speed switch: multi-row routing, including
plain prompt prefill above three rows, runs the half-precision GEMM. It costs
**2.35 ms/step** in decode (49 rocBLAS `mm` calls at 47.9 µs each) and is required for
DFlash2's quality gate, so it is not free to reclaim. Do not set it under tensor
parallelism — replicated routing depends on the deterministic projection, and displaced
ranks could select different experts. The value is read per call, but each layer latches
at its first routing call, so set it before loading.


---

## Running the deployment

The model runs from a pinned tree with the compiled extension on `PYTHONPATH`, one process
per GPU. The server script (`ornith_serve.py`) and its frozen runtime live in the
deployment bundle, **not in this repository** — this repo carries the extension source and
the kernel/router changes:

```bash
EXL3_ROUTER_DET=0 \
  python -P <bundle>/ornith_serve.py \
    --model  /path/Ornith-1.5-122B-EXL3-EAQv1-4-bpw \
    --draft-model /path/Ornith-1.5-122B-DFlash2 \
    --ctx 262144 --port 18161 --host 0.0.0.0 \
    --served-name ornith-1.5-122b-exl3
```

`--draft-model` selects DFlash2. Omit it and pass `--draft 2` for the MTP drafter.
`EXL3_ROUTER_DET=0` is **required** with DFlash2, otherwise the 0.25 quality gate fails as
described above.

**The draft depth is not configurable from the launcher.** `NDT` in the launch script is
vestigial — it is never passed to the server. The depth comes from the DFlash2 drafter's
own `default_draft_size` cap, computed as `config.block_size - 1` in
`exllamav3/architecture/dflash2.py:223`, and consumed at `ornith_serve.py` with no
override. Changing `block_size` is **not** a safe substitute: it is the drafter's trained
native draft length (`native_draft_len = config.block_size`) and gates the conv kernel, so
shortening it risks breaking the drafter rather than just shortening the verify. Exposing
a proper depth override is an open item — see [Known headroom](#known-headroom).

---

## Measured performance

All numbers below were measured on this box (single gfx1151, ctx 262144, server resident),
using the deployed build, `ArgmaxSampler`, fresh salted prompts, and generation forced to an
exact token count so a stray EOS cannot inflate the rate. Full method and raw tables:
[docs/gfx1151-ornith122-baseline-20261029.md](docs/gfx1151-ornith122-baseline-20261029.md).

### Prefill

| workload | t/s |
|---|---|
| 3–4k prompt | 682 |
| **32768-token prompt** | **612.7** (53.48 s) |

### Decode — draft depth matters enormously

400-token prompt, 64 tokens forced, one process, same harness:

| draft depth | tok/round | ms/round | **t/s** |
|---|---|---|---|
| ndt 15 (deployment default) | 6.30 | 366.7 | **17.18** |
| **ndt 8** | 5.73 | 278.1 | **20.59** |
| ndt 4 | 3.94 | 230.1 | 17.11 |
| ndt 2 | 2.86 | 139.2 | 20.57 |
| ndt 0 (plain AR, no speculation) | 1.00 | 60.5 | 16.5 |

**ndt 15 accepts *more* tokens per round yet costs 32 % more per round.** With 256 experts
and top-8, `k` verify rows touch `256·(1−(1−8/256)^k)` distinct experts per layer — **102 at
k=16 versus 64 at k=9** — so a deeper verify inflates routed-expert traffic faster than it
adds tokens. Measured, speculation buys much less than it costs beyond a shallow window:
ndt 8 and ndt 2 reach **1.25×** over plain AR (20.6 vs 16.5 t/s), while ndt 15 manages only
1.04× — which is why **ndt 8 and ndt 2 both beat the deployed ndt 15 by ~20 %**.

Quality at ndt 8 is equivalent to ndt 15 on the reference prompt: both first diverge from AR
greedy at character 142 of 229, with the same substitution ("blocks" → "sub-blocks").

### The model's cost structure

| | |
|---|---|
| Active parameters per token | ~9.0 B (48 layers, 256 experts top-8, hidden 3072) |
| Compute | ~18 GFLOP/token |
| Weights read per AR token | **5.27 GB** (routed experts 1.84 GB, lm_head 0.57 GB at 6 bpw, rest dense/GDN) |
| Measured DRAM read ceiling at the 15–24 MB operand sizes | **189–224 GB/s** |
| Measured dense fp16 GEMM peak | **32–38 TFLOPS** |
| GPU compute units | **40** (torch reports `multi_processor_count = 20` — that is WGPs) |

### Where the time goes

Device-side kernel breakdown, prefill 8192 tokens (13.02 s GPU): `exl3_moe_kernel` **36 %**
(12.7 TFLOPS), dense `exl3_wmma_gemm_kernel*` ~22 %, FLA/GDN chunk kernels ~16 %,
`_paged_attn_prefill` 6.6 %, `reconstruct_had` 2.3 %.

AR decode (42.6 ms/step GPU, 124 GB/s effective): dense GEMV family ~38 %, MoE
`exl3_moe_dec_dot` ~23.5 %, lm_head 6.4 % (already at 212 GB/s, at the memory ceiling),
router 5.5 %, GDN 3.9 %.

**The GEMV is instruction-issue bound, not bandwidth bound.** Removing the dequant and dot
while keeping memory traffic byte-identical (a diagnostic build, 64 tokens forced, both
configs in one process) gives:

| build | 64 tokens | t/s |
|---|---|---|
| load-only (arithmetic removed) | 2.20 s | **28.63** |
| baseline | 3.82 s | **16.51** |

That 1.73× is why every memory-side fix failed: "124 GB/s" was never a bandwidth figure.
It is an **upper bound that requires the algorithm to vanish** — the 5-bit `cb=mul1` decode
is already hand-written inline asm (`v_pk_mul_lo_u16`, `v_mad_u32_u16`, `v_lop3`,
`v_pk_add_f16`), ~6–7 instructions per weight of which only ~1 is the FMA, and EXL3 is
QTIP-derived so the hash *is* the codebook and cannot be deleted.


---

## Ceilings: why 60 t/s decode and 1500 t/s prefill are not reachable here

These are not tuning gaps. Both follow from the model and the part.

**Decode 60 t/s needs 316 GB/s.** 5.27 GB/token × 60 = 316 GB/s, against a **256 GB/s
theoretical** LPDDR5X limit, before any verification cost. Even with ideal speculation
(1.56×) it needs 203 GB/s — above the 178–224 GB/s the memory system actually delivers.
Best case with perfect speculation and perfect bandwidth: **~57 t/s**.

**Prefill 1500 t/s needs 27 TFLOPS.** 1500 × 18 GFLOP = 27 TFLOPS from a kernel that must
also trellis-dequantise 4-bit weights, against a measured **32–38 TFLOPS dense fp16 peak**.
Summing measured device time at 100 % of peak GEMM with GDN and attention left exactly as
slow as they are today gives `3.88 + 2.05 + 0.85 = 6.78 s → 1208 t/s`; at the 4096³ peak,
1100 t/s. Deleting the MoE kernel from the timeline entirely still only yields 987 t/s.
Prefill is compute-bound on active FLOPs, so **bitrate cannot help it**.

**Practical ceilings: decode ~18–19 t/s AR (~21 with best speculation), prefill ~750–800 t/s.**

### Bubbles are not the problem

Per-kernel timestamps from chrome traces (26,314 kernels decode, 14,389 prefill):

| | span | busy | idle | median inter-kernel gap |
|---|---|---|---|---|
| decode | 1012.7 ms | 922.8 ms | **8.9 %** | **2 µs** |
| prefill | 6598.3 ms | 6342.7 ms | **3.9 %** | **2 µs** |

Kernels launch back-to-back; fusion and graph capture have almost nothing to recover. The
few large gaps follow host-side work (`copyBuffer`, `elementwise_kernel_manual_unroll`,
`_scatter_gather_elementwise`, `kernelHistogram1D`), not launch overhead — which is why
`EXL3_GRAPHS=1` regresses on ROCm.

---

## Rejected experiments — do not repeat these

Every entry was **measured** on this deployment. Baseline: AR 16.5–17.2 t/s,
DFlash2 ndt8 20.6–21.2 t/s, prefill 32k 612.7 t/s.

| Candidate | Result |
|---|---|
| Fold `exl3_gemv_mr_had_in_single` into the dot kernel | **7×–200× loss.** The rotation is amortised *intra*-launch across `grid.x` blocks; folding makes each block redo it (188× at tpb=2). It also publishes the `GP_gemm_*` graph patch pointers. |
| "1 of 48 MoE layers on `exl3_moe_coop`" | **Not reproducible.** Route census (3168 records): 48/48 `moe_decode_rdna`, zero coop at bsz=1. |
| `EXL3_MGEMV_MAX_M=16` | No gain (ndt15 456 vs 439 ms/round). |
| `EXL3_GRAPHS=1` on ROCm | **Regresses.** ndt15 439→355 ms but AR 16.5→13.8 t/s. |
| `EXL3_GEMV_MR_M1=0` (~180 fewer launches, bit-identical) | No gain (AR 16.26 vs 16.52–17.10). The fused LDS prologue costs more occupancy than the launch saves. |
| `EXL3_GEMV_TILES_TMAX` 2→4 | ~0.3 % by arithmetic; not built. `A_had` at m=1 is 6 KB vs 23.6 MB of weights. |
| Packed-weight coalescing / wider loads | **Not the cause.** 4.00× sector *request* amplification but **1.00× fetched bytes** (zero DRAM over-fetch), ~4 % of L1 sector pipeline. Wider loads over-read the tile. |
| Per-thread MLP: U 1→2 for 5-bit/T=2 | **Bit-identical, no gain** (numerics EQUIVALENT; 20.78 vs 20.72–21.22 t/s). 0 spills, VGPR 55→73, occupancy unchanged — the cliff is at 95 VGPRs, not the assumed 64. |
| `EXL3_GEMV_LDS=1` (LDS unswizzle core) | **−9.7 %** (4198.0 vs 3789.5 ms/round). |
| `EXL3_ROCM_GEMV_TILES=0` (direct core) | **−3.4 %** (3922.7 vs 3789.5 ms/round). |
| Fusion / graph capture for launch bubbles | ≤9 % available, mostly host-side and not fusable. |

---

## Known headroom

The one measured, unexploited win is **draft depth: ndt 8 or ndt 2 beat the deployed
ndt 15 by ~20 %** with equivalent divergence from AR greedy on the reference prompt.
Enabling it requires decoupling verify depth from the drafter's trained `block_size` —
either a new cap/CLI flag in `exllamav3/architecture/dflash2.py` plus a rebuild, or an
override in the serve bundle. Neither is a config toggle today.

Beyond that, the remaining slack is ALU scheduling in the 5-bit decode, estimated
**+10–20 %**, not the 1.73× that the load-only A/B shows is theoretically there.

---

## Diagnostics

- **Device tracing is off by default.** `exllamav3/__init__.py` sets
  `ROCPROFILER_REGISTER_ENABLED=0`, which silently disables **all** device tracing —
  `torch.profiler` *and* `rocprofv3` return zero records. Export
  `ROCPROFILER_REGISTER_ENABLED=1` before running to get device events. This makes the
  deployment look unprofilable otherwise.
- **Route trace.** `EXL3_ROUTE_TRACE=1` records the dispatch route of every
  `exl3_gemm` / `exl3_mgemm` call and of the routed-expert tiers, with shape, bitrate,
  codebook and flags. Use it before attributing any numerical or performance difference to
  a kernel family — it is how the m=1 vs m=16 tier split was found.
- **`EXL3_GEMV_LDS`, `EXL3_ROCM_GEMV_TILES`, `EXL3_GEMV_MR_M1`, `EXL3_MGEMV_MAX_M`,
  `EXL3_GEMV_TILES_TMAX`** are all runtime or compile-time core-selection switches. Their
  measured outcomes are in the rejected register above — the defaults are the fastest
  configurations found.

---

## Tests added by this fork

`tests/kernels/gemm/test_exl3_gemv_rowgroups.py` (+175),
`test_exl3_mgemm_rowgroups.py` (+172), `test_exl3_route_trace.py` (+194),
`test_exact_linear.py` (+136), plus a two-line change in `test_exl3_gemv.py`.

---

## Acknowledgements

Upstream ExLlamaV3 and its community — see the
[upstream project](https://github.com/turboderp-org/exllamav3) for the full list. This fork
builds on QTIP-derived EXL3 quantization, Flash Linear Attention (vendored under
`exllamav3/vendor/fla`), PyTorch and rocBLAS.

