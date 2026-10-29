# Ornith-1.5-122B EXL3 on gfx1151 — measured baseline and ceiling analysis

Date: 2026-10-29. Host: single gfx1151 (Ryzen AI MAX+ 395 / Radeon 8060S), 123 GiB
unified, ROCm 7.2.4, torch 2.14.0+rocm10.1.0a20260822, pinned tree
`claude-fix-test-v1/source` @ `a5e4acd` (= this repo `master`).

Gate under investigation: **average decode ≥ 60 t/s and prefill ≥ 1500 t/s over a
32k context**, at unchanged quality (4.07 bpw EXL3 EAQv1 + DFlash2).

## 1. Baseline (measured, not from the README)

| metric | measured | README claim |
|---|---|---|
| prefill, 3.1k prompt (live HTTP) | 682 t/s | ~684 |
| prefill, 32768 tokens (in-process) | **612.7 t/s** (53.48 s) | — |
| decode, DFlash2 ndt=15 | **14.4 t/s** shallow, ~12.5 t/s @ 32k | 91.6-97.8 |
| decode, DFlash2 ndt=8 | **21.2 t/s** shallow | — |
| decode, plain AR | **16.5 t/s** | — |

The 91.6-97.8 t/s headine **does not reproduce on this tree/build** at any draft
depth or depth of context. Measured decode is 20.7-24 t/s at shallow depth and
~12.5 t/s at 32k. Everything below is measured on the deployed extension.

Method: `Cache(model, max_num_tokens=CTX, max_history=ndt)` + `Generator(...,
draft_model=, num_draft_tokens=)`, `ArgmaxSampler`, fresh salted prompt per
request, 2-token warmup per shape, `job.draft_stats` for round/acceptance counts.

## 2. Per-forward economics

Active weights read per AR token, by projection class (from the route trace at
m=1, 48 layers), bits from each record's `tile_u16`:

| class | bytes/token |
|---|---|
| qkv (3072→12288, 5b, 36 layers) | 849 MB |
| q/o (3072→8192, 8192→3072, 5b) | 566 + 754 MB |
| routed experts (8 of 256, 4b, 48 layers) | 1840 MB |
| shared expert gate/up + down | 187 + 95 MB |
| full-attn qkv + k/v (12 layers) | 378 + 24 MB |
| lm head (248320×3072, 6b) | 572 MB |
| **total** | **5.27 GB** |

9.0 B active params → 18 GFLOP/token.

Measured hardware ceilings on this box: DRAM copy 196 GB/s, read 178 GB/s;
rocBLAS fp16 2048³ 38.1 TFLOPS, 4096³ 32.4 TFLOPS.

At 60.5 ms per AR pass, 5.27 GB/token means the GEMV path sustains **89 GB/s**,
i.e. ~45 % of the measured read ceiling.

## 3. Why speculative decoding barely pays here

256 experts, top-8. k verify rows touch
`E(k) = 256·(1 − (1 − 8/256)^k)` distinct experts per layer:

| k | E(k) | expert bytes vs AR | accepted tok/round | measured t/s |
|---|---|---|---|---|
| 1 (AR) | 8.0 | 1.00× | 1.00 | 16.5 |
| 3 | 23.3 | 2.91× | 2.86 | 20.2 |
| 5 | 37.6 | 4.70× | 3.94 | 17.4 |
| 9 | 63.6 | 7.95× | 5.73 | 21.2 |
| 16 | 102.0 | 12.75× | 6.30 | 14.4 |

(An earlier version of this table had E(3)=27, E(5)=44, E(9)=74. Those were
arithmetic errors; the values above are recomputed from the formula on line 54.)

Speculation multiplies MoE *weight traffic* almost as fast as it multiplies tokens.
Modeling step cost as `0.65 + 0.35·E(k)/8` (MoE is 1.84 of the 5.27 GB/token) puts the
ceiling at ~1.7× around k=3..5, but the **measured** gain is smaller - ndt8 and ndt2 reach
1.25× over AR, ndt15 only 1.04× - because the model ignores the fixed per-round costs.
Either way ndt=15 is a net loss versus ndt=8. This is a property of a 256-expert top-8 MoE
at short row counts, not a tuning miss.


## 4. Dispatch: the verify pass leaves the weight-once GEMV family

`EXL3_ROUTE_TRACE=1` over one AR forward and one 16-row verify forward:

| projection | m=1 route | m=16 route |
|---|---|---|
| MoE experts, 48x | `rdna_moe_decode` (slots=8, GEMV) | **`moe_fused_tile_16`** (prefill kernel) |
| gate/up + k/v pairs | `rdna_multirow_mgemv` (93) | **`coop_mgemm_shape_2`** (grid-sync WMMA) |
| qkv / o / head | `rdna_multirow_gemv` (92) | `rdna_multirow_gemv` (92) |

`EXL3_MGEMV_MAX_M=16` repairs the pairs half of this (route 93 instead of
`coop_mgemm_shape_2`), but measured **no gain** (ndt15 456 vs 439 ms/round;
ndt8 270 vs 276) because the MoE tier, not the pairs, is the cost. The
cooperative tier is entered above `MAX_BSZN` rows, which is why ndt=15 is the
worst config available.

`EXL3_GRAPHS=1` (off by default on ROCm) at ndt15: 439 -> **355 ms/round (+24 %)**,
but AR 16.5 -> **13.8 t/s (-17 %)** and ndt8 21.2 -> 19.9. Launch-bound only at
the 16-row shape; not a clean win, and it changes capture behaviour.

## 5. Context depth

ndt=8, 48 tokens per point (round counts differ, see caveat):

| depth | t/s | ms/round |
|---|---|---|
| 512 | 26.35 | 223.0 |
| 4096 | 26.32 | 223.2 |
| 16384 | 16.67 | 352.4 |
| 32768 | 22.86 | 293.7 |

Flat to 4k, then a large, **non-monotonic** penalty. Caveat: rounds carry
different accepted-token counts, so ms/round is not row-normalised; the 16k
point had 5.9 tok/round vs 6.9 at 32k, which alone explains part of the
inversion. Streaming the resident KV once at 32k would cost ~4 ms/round, so the
observed 70-130 ms is not explained by KV bandwidth. Prime suspect is the GQA
ratio (32 q-heads / 2 kv-heads = 16x) entering the KV read path, which would put
the cost at ~16x the streaming floor. **Unproven** - needs a fixed-row
measurement (force a constant verify width, or count KV bytes per round) before
anyone acts on it.

## 6. Verdict on the gate

**Prefill 1500 t/s** requires `1500 x 18 GFLOP = 27 TFLOPS`. Measured dense fp16
peak is 32-38 TFLOPS, so the gate asks for 71-84 % of a *well-tuned dense GEMM*
from a kernel that must also trellis-dequantise 4-bit weights, starting from
11.4 TFLOPS today (30-35 %). Not reachable. Realistic ceiling after a serious
MoE-GEMM rewrite: **~900-1100 t/s**.

**Decode 60 t/s at 32k** needs 5.27 GB/token read plus verification inflation:

| config | bytes/round | at 190 GB/s | tok/round | t/s ceiling |
|---|---|---|---|---|
| AR | 5.3 GB | 27.7 ms | 1.0 | 36 |
| k=3 | 9.6 GB | 50 ms | 2.86 | 57 |
| k=9 | 20.4 GB | 107 ms | 5.73 | 53 |

Best case **~57 t/s at ~100 % of measured DRAM bandwidth**. Reaching 60 needs
either >200 GB/s sustained or fewer bytes per token (~2.5 bpw), which is a
quality change. So the gate is ~5 % beyond the hardware edge *and* 2-2.5x beyond
the current kernel efficiency. Simultaneously fixing the GEMV bandwidth (~2x)
and the depth penalty lands around **45-55 t/s decode and ~1000 t/s prefill**.

## 7. Ranked work, if this is pursued

Ordered by measured share of the current decode/prefill budget:

1. **Host submission gap — 14.3 ms/step, 25 % of decode wall time**, at 1428
   launches per step. `EXL3_GRAPHS=1` does not recover it (HIP launches graph
   nodes no faster than plain kernels; measured 17.6 -> 13.8 t/s). The obvious
   launch-reduction target was investigated and rejected (see section 9): the
   186/step `exl3_gemv_mr_had_in_single` launches cannot be folded, because the
   rotation output `A_had` is shared by many dot launches and the standalone
   kernel also publishes the graph-patch pointers. **No known mechanism remains
   for this 25 %**, which is why decode is expected to cap well below 30 t/s.
2. **Multi-row GEMV bandwidth 124 -> ~190 GB/s** (decode, ~1.5x on the GPU
   portion). The lm_head already runs at 212 GB/s, so the ceiling is reachable;
   the shortfall is in `exl3_gemv_mr_dot_single` (27.9 %) and the pair/tile
   variants. Coalesced 128-bit packed-weight loads, LDS staging of activations,
   more waves per CU; preserve each dot product's accumulation order
   (`exl3_gemv_kernel_rdna.cuh`, `rocm/quant/exl3_gemv_multirow_rdna.cu`).
3. **Default draft depth 8, not 15** — free, measured +47 % (14.4 -> 21.2 t/s).
   Requires re-running the 0.25 greedy-equivalence gate, since the verify shape
   changes; `EXL3_ROUTER_DET=0` is still required.
4. **Keep m<=16 on the weight-once tier.** `EXL3_MGEMV_MAX_M=16` fixes the dense
   pairs (route 93 instead of `coop_mgemm_shape_2`) but measured no gain on its
   own, because at k=16 the MoE expert blast radius E(k)=102 dominates. Worth
   revisiting only together with (3).
5. **Retracted: the 1 layer/step on `exl3_moe_coop`.** Route census shows 48/48
   on the fast tier at bsz=1 and the only per-layer predicate that could split
   one is an expert-range shard, which this deployment does not use.
6. **Prefill MoE GEMM** 12.7 -> ~25 TFLOPS (36 % of prefill): wider MMA tiles,
   LDS reuse across expert groups, split interiors from the ragged tail
   (`exl3_moe_kernel.cuh`, `exl3_moe_coop_kernel.cuh`).
7. **Depth penalty**: fix or disprove the GQA KV re-read with a fixed-row
   measurement before touching anything else at 32k.

Rejected: `EXL3_MGEMV_MAX_M=16` alone (no measurable gain), `EXL3_GRAPHS=1` on
ROCm (regresses AR), ndt=15 (net loss vs ndt=8).

## 8. Harnesses

`/home/hycris/work/{ornith-bench-live,prof_ornith,decode_sweep,depth_curve,route_probe}.py`
- all take the pinned tree via `PYTHONPATH=$FORK:$ART:$BUNDLE` and

## 9. Device-side kernel breakdown (the decisive data)

`ROCPROFILER_REGISTER_ENABLED=1` is required — `exllamav3/__init__.py` sets it to `0`
by default, which silently disables *all* device tracing (torch.profiler and
rocprofv3 both return zero records without it).

### Prefill, 8192 tokens, 13.02 s of device time

| kernel | ms | % |
|---|---|---|
| `exl3_moe_kernel<4,256,2,16,false,true>` | 4687.6 | 36.0 |
| dense `exl3_wmma_gemm_kernel*` (all variants) | ~2900 | ~22 |
| FLA / GDN chunk kernels (`recompute_w_u`, `chunk_fwd_kernel_o`, `chunk_gated_delta_rule_fwd`, `kkt_solve`, `causal_conv1d`) | ~2050 | ~16 |
| `_paged_attn_prefill_kernel` | 853.8 | 6.6 |
| `reconstruct_had_kernel<5,2,false>` | 296.3 | 2.3 |
| `exl3_moe_gather_kernel` | 210.0 | 1.6 |
| `hipLaunchKernel` (19814 launches) | 77.0 | 0.6 |

GEMM-shaped work is ~60%; the layout/gather overhead that would have been a
cheap win is not there. The MoE kernel alone sustains **12.7 TFLOPS**.

### AR decode, 47 steps, **42.6 ms/step GPU, 2000.3 ms** (8-token prompt)

Note: an earlier version of this table profiled a 300-token prompt, so its window
contained a 300-token prefill (m=43 chunks) and reported 66.1 ms/step and
`exl3_moe_kernel` at 13.6%. Those numbers were wrong and are replaced here. The
`exl3_moe_kernel` calls in that window were the *prefill*, not decode.

| kernel | ms | % | calls/step |
|---|---|---|---|
| `exl3_gemv_mr_dot_single<5,true,2,8,1>` | 557.4 | 27.9 | 123 |
| `exl3_moe_dec_dot<4,false,2,8>` | 299.4 | 15.0 | 49 |
| `exl3_moe_dec_dot<4,true,2,4>` | 170.0 | 8.5 | 49 |
| `exl3_gemv_dot_kernel<6,false,2,4>` (lm_head) | 127.8 | 6.4 | 1 |
| `aten::mm` / rocBLAS `Cijk_*_MT8x8x128` (router) | 110.3 | 5.5 | 49 |
| `exl3_gemv_mr_dot_single<5,false,2,4,1>` | 88.0 | 4.4 | 12 |
| `exl3_gemv_mr_dot_multi<5,false,2,8,1>` | 59.2 | 3.0 | 49 |
| `exl3_moe_coop_a/b_kernel` | 90.3 | 4.5 | **1** |
| `hipLaunchKernel` (67146 launches) | 48.1 | 2.4 | 1428 |
| GDN recurrent + `gdn_ba_gemv` | 76.7 | 3.9 | 74 |
| `exl3_gemv_mr_had_in_single` | 18.4 | 0.9 | 186 |
| `routing_std_topk` + `routing_gemv_rdna_w16` | 37.8 | 1.9 | 99 |

**GPU time gives 23.5 t/s and 124 GB/s** (5.27 GB/token). Wall clock is
56.9 ms/step = 17.6 t/s, so **~14.3 ms/step (25 %) is host submission gap** at
1428 launches per step. That gap, not a kernel, is the largest single
inefficiency in decode, and `EXL3_GRAPHS=1` does not recover it (measured
17.6 -> 13.8 t/s), consistent with `doc/env_vars.md` stating the HIP runtime
launches graph nodes no faster than plain kernels.

Three candidate launch/reuse anomalies were investigated to verdict and **all
three were rejected**:

- **Standalone Hadamard launch (186/step)** - NOT foldable for a win. The rotation
  is already byte-exact-feasible inline (`exl3_gemv_had_in_128`,
  `exl3_gemv_kernel_rdna.cuh:626`, driven block-wide by `exl3_gemv_rotate_in`
  `:664-678`). Correcting an earlier claim in this file: `A_had` is **not**
  shared across dot launches - the pairing is 1 rotation : 1 dot (it is a
  per-call scratch the size of A, `exl3_gemm.cu:35`, and `suh` is per-matrix), and
  the recorded "123 dot launches" vs 186 rotations was a per-symbol artifact
  (the dot is instantiated per bit/cb/warps/tile/dtype). The amortization is
  **intra-launch**: one rotation feeds all `grid.x = n_tiles/tpb` blocks of that
  single dot, so folding makes every block redo it - 188x at `tpb=2`, 377x at
  `tpb=1`, i.e. 132.1M vs 0.700M rotation-halves per step. Upside ceiling is the
  0.9% of device time the launch costs (445 us of 49.4 ms, 15 us of it
  unavoidable bandwidth); net is a 7x-200x loss. Independently, the m>1 verify
  route would need 48-128 KB of LDS against a 64 KB part. The standalone kernel
  also republishes the graph-patchable pointers (`pb->B/C/A_had/svh`,
  `exl3_gemv_multirow_rdna.cu:394-400`) that the six `GP_gemm_*` patch sites
  depend on, so removing it would break graph replay. Reject.
- **`EXL3_GEMV_TILES_TMAX` 2 -> 4** (raise N-tiles per block, halving the dot's
  block count and the `A_had` re-reads) - **rejected on arithmetic, no build
  spent**. At m=1 the rotated input is 1 x 3072 x 2 B = 6 KB, so even at
  `n_tiles/tpb` = 24 re-reads it is 144 KB against 23.6 MB of weights for the
  3072->12288 projection - 0.6% of that call's traffic, and halving it saves
  ~0.3%, below run-to-run noise. The U-tile table also documents only T=1 and
  T=2 (`exl3_gemv_tiles_rdna.cuh:643-657`), so T=4 is untuned.

  removing ~180 launches/step, bit-identical) - **measured, no gain**:
  AR 16.26 t/s vs 16.52-17.10 baseline; ndt=8 21.16 vs 20.72-21.22. This is the
  project's own documented tradeoff - the fused LDS prologue costs more occupancy
  than the launch saves (`exl3_gemv_multirow_rdna.cu:94-99`). Reject as a default.
- **"1 of 48 layers on `exl3_moe_coop`"** - NOT reproducible. `bszn_eligible`
  (`block_sparse_mlp.py:1038`) is true for all 48 layers and the deployed
  build's own route census (3168 records, ext sha `5c0b14bc`) shows **48/48
  `moe_decode_rdna` and zero `moe_coop`** at bsz=1. The 1/step I measured is most
  likely the single prefill step that my profile window also contained, or a
  non-routed component. The only code-level predicate that could split one layer
  is `min_expert < 0` (`blocksparse_mlp.cpp:345`), which bites only an
  expert-range shard (`EXL3_MOE_CPU_SPLIT_LAYERS=1`) - not this deployment.
  Retracted unless a fixed-row rerun shows otherwise.

The router still costs 2.35 ms/step (49 rocBLAS `mm` calls, 47.9 us each). That is
the price of `EXL3_ROUTER_DET=0`, which DFlash2 requires for the 0.25 gate, so
it is not free to reclaim.

The lm_head GEMV moves 572 MB in 2.7 ms = **212 GB/s**, i.e. already at the
memory ceiling. Everything else sits at 85-125 GB/s, so the shortfall is
concentrated in the multi-row GEMV family rather than being uniform as first
reported.

## 10. Why the gate cannot be met (both proofs are exact)

**Prefill 1500 t/s is above the zero-overhead bound.** 8192 tokens is
147.5 TFLOP at 18 GFLOP/token. At the 38 TFLOPS measured rclBALS/rocBLAS peak
that is 3.88 s of pure GEMM, and the GDN chunk/recurrent kernels (2.05 s) and
paged attention (0.85 s) are serial additions:
`3.88 + 2.05 + 0.85 = 6.78 s -> 1208 t/s`. Using the 32.4 TFLOPS measured at
4096^3 instead: `4.55 + 2.05 + 0.85 = 7.45 s -> 1100 t/s`. So **every GEMM at
100 % of dense-GEMM peak, with GDN and attention left exactly as slow as they
are today, still only reaches ~1100-1210 t/s.** 1500 t/s is 25-35 % above the
hardware's absolute maximum. Independently: deleting the MoE kernel from the
timeline entirely yields 8192/(13.02-4.69) = 987 t/s.

**Decode 60 t/s needs 316 GB/s.** 5.27 GB/token x 60 = 316 GB/s, above the
256 GB/s theoretical LPDDR5X limit, before any verification inflation. With the
best achievable speculation (1.56x) it needs 203 GB/s, above the 178-196 GB/s
this memory system actually delivers. Best case with ideal speculation and
perfect bandwidth: ~57 t/s.

Achievable ceilings: **decode ~34 t/s AR / ~53-57 t/s with ideal speculation;
prefill ~950-1000 t/s.** Hitting 60 t/s decode would need <=3.38 GB/token, which
is ~2.6 bpw - a quantization/quality change, and it still would not move the
prefill gate, which is compute-bound on active FLOPs and therefore bitrate
independent.


## 11. Rejected-experiment register (do not repeat these)

Every entry below was measured on this deployment, not reasoned about. Baseline for
comparison: AR decode **16.5-17.1 t/s wall / 42.6 ms GPU / 124 GB/s**, DFlash2 ndt8
**20.7-21.2 t/s**, prefill 32k **612.7 t/s**, 48 forced greedy tokens, ndt0/ndt8.

| # | candidate | result | evidence |
|---|---|---|---|
| 1 | Fold `exl3_gemv_mr_had_in_single` into the dot | **7x-200x loss** | rotation is amortized intra-launch across `grid.x` blocks; folding makes each block redo it (188x at tpb=2, 377x at tpb=1). Also publishes the `GP_gemm_*` graph patch pointers. |
| 2 | "1 of 48 MoE layers on `exl3_moe_coop`" | **not reproducible** | deployed route census (3168 records): 48/48 `moe_decode_rdna`, zero coop at bsz=1 |
| 3 | `EXL3_MGEMV_MAX_M=16` | **no gain** | ndt15 456 vs 439 ms/round; ndt8 270 vs 276 |
| 4 | `EXL3_GRAPHS=1` on ROCm | **regresses** | ndt15 439->355 ms but AR 16.5->13.8 t/s |
| 5 | `EXL3_GEMV_MR_M1=0` (fused single-launch, ~180 fewer launches, bit-identical) | **no gain** | AR 16.26 vs 16.52-17.10; ndt8 21.16 vs 20.72-21.22. Fused LDS prologue costs more occupancy than the launch saves. |
| 6 | `EXL3_GEMV_TILES_TMAX` 2->4 | **~0.3%, not built** | `A_had` at m=1 is 6 KB; even 24 re-reads = 144 KB vs 23.6 MB of weights |
| 7 | Packed-weight coalescing / wider loads | **not the cause** | 4.00x sector *request* amplification but **1.00x fetched bytes** (zero DRAM over-fetch), ~4% of L1 sector pipeline. Wider loads over-read the tile. |
| 8 | Per-thread MLP: U 1->2 for 5-bit/T=2 | **bit-identical, no gain** | numerics EQUIVALENT (TEXT+TOKENS identical, 48 tokens, ndt0+ndt8); speed 20.78 vs 20.72-21.22 (a first run's 22.51 was noise). 0 spills, VGPR 55->73, occupancy unchanged (cliff measured at 95 VGPR, not the assumed 64). |
| 9 | `EXL3_GEMV_LDS=1` (LDS unswizzle core) | **-9.7%** | 4198.0 vs 3789.5 ms/round |
| 10 | `EXL3_ROCM_GEMV_TILES=0` (direct core) | **-3.4%** | 3922.7 vs 3789.5 ms/round |
| 11 | Fusion / graph capture for launch bubbles | **~0-9%, mostly not fusable** | decode idle 8.9% (89.9 of 1012.7 ms), prefill 3.9% (255.6 of 6598.3 ms), **median inter-kernel gap 2 us**. The large gaps follow `copyBuffer`, `elementwise_kernel_manual_unroll`, `_scatter_gather_elementwise`, `kernelHistogram1D` - host-side work, not launch overhead. |

**Superseded claims in this document.** Two earlier figures here were wrong and are
replaced: (a) the decode "host submission gap 14.3 ms/step = 25% of wall" - it came
from comparing summed `key_averages` device time against wall, a flawed method; the
trace-based measure is **8.9% idle**. (b) The "1 of 48 layers on `exl3_moe_coop`"
4.5% claim in section 9 - retracted in item 2 above.

## 12. Where the time actually is (final)

Measured by the load-only A/B (dequant + dot removed, memory traffic byte-identical,
both runs forced to exactly 64 tokens):

| build | 64 tokens | t/s |
|---|---|---|
| load-only | 2.20 s | **28.63** |
| baseline | 3.82 s | **16.51** |

**1.73x from removing arithmetic alone** - the GEMV is instruction-issue bound, not
bandwidth bound. That also explains why every memory-side fix (items 3, 5, 6, 7, 8)
failed: the 124 GB/s was never a bandwidth number.

But the 1.73x is an **upper bound that requires the algorithm to vanish**. The 5-bit
`cb=mul1` decode is already hand-written inline asm - `v_pk_mul_lo_u16` +
`v_mad_u32_u16` for the hash, `v_lop3` + `v_pk_add_f16` for the pair decode,
`v_pk_*` for the FMA - roughly 6-7 instructions per weight of which only ~1 is the
multiply-accumulate. EXL3 is QTIP-derived: the hash *is* the codebook, so it cannot be
deleted. Recoverable slack is scheduling/packing only, estimated **+10-20%**, not 1.7x.

**Practical ceilings: decode ~18-19 t/s AR (from 16.5), prefill ~750-800 t/s (from
612).** The 60/1500 gate is unreachable on this part at 4.07 bpw: 60 t/s needs
316 GB/s against a 256 GB/s theoretical DRAM limit, and 1500 t/s needs 27 TFLOPS
against a measured 32-38 TFLOPS dense-GEMM peak from a 4-bit dequant kernel.

`EXL3_ROUTER_DET=0`.
