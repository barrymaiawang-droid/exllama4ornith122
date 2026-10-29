# perf-headroom: five decode-side changes from the local session, merged 2026-10-29

Source branch `claude/perf-headroom` (head `7e724f5`), written on a host without gfx1151 or
torch and **never compiled or run**. Merged into `master` on top of the upstream v1.6.5 sync
(`docs/upstream-sync-v1.6.5-20261029.md`) without conflicts; the two generator changes sit
side by side (see §3). Everything below is for the local agent on the pinned box.

## 1. What the five commits do

| commit | change | default behaviour |
|---|---|---|
| `bcce670` | Router: with `EXL3_ROUTER_DET=0` the displaced int8 gate is an empty tensor that still `has_value()`, so 4..16-row router calls fell to the rocBLAS hgemm. Now 1..16 rows take the RDNA router GEMV (9..16 as two 8-row launches); each row keeps the single-row FMA chain, so verify-window routing scores are bit-identical to AR's. Adds `tests/kernels/routing/test_router_rows_exact.py`. | changes routing for the `EXL3_ROUTER_DET=0` deployment (intended); `EXL3_ROUTER_DET` unset is unchanged |
| `8dcf7b2` | `EXL3_DRAFT_TOKENS`, `EXL3_DYNAMIC_DRAFT`, `EXL3_DRAFT_CONFIDENCE`: environment overrides of the Generator's draft depth / dynamic window / confidence target, so the frozen serve bundle can choose them. DFlash2 depth is capped at the drafter's `default_draft_size`. | unchanged when unset |
| `bd26d86` | `EXL3_DRAFT_COST="rows:ms,..."`: cost-aware dynamic window; each round picks the window maximizing expected accepted tokens per ms from the calibrator's per-position estimates. Adds `tests/generator/test_cost_aware_window.py`. | opt-in |
| `d942d1c` | `EXL3_ATTN_SPLIT_TARGET`: flash-decoding split target (programs per multiprocessor) for the paged decode attention; torch counts WGPs on RDNA so the historical 2 gives few splits at long context. | unchanged (2) |
| `7e724f5` | `modules/moe_route_sort.py`: one-launch Triton expert grouping for <= 128 assignments. **Not wired in**; `EXL3_MOE_FAST_SORT` reserved. | inert |

Review notes (source only): the router kernel's per-row chain is independent of the template
row count and the m = 1 16-byte-load kernel walks the same columns in the same order, so the
bit-identity claim holds by construction; `EXL3_DRAFT_TOKENS=0` cannot disable drafting (falls
back to the default); the cost-aware window picks 1 draft token at constant per-position
confidence <= 0.8 and the full window at >= 0.95 with the table in `doc/env_vars.md`.

## 2. How to run it

```
git checkout master && git pull --ff-only
# rebuild the extension (routing.cu changed; the upstream sync also added generator/staging.cpp)
python -c "from exllamav3.ext import exllamav3_ext as e; print(hasattr(e, 'generator_stage_decode'))"   # True

# 1. router: 2..16-row scores, selections and weights equal the single-row path
pytest tests/kernels/routing/test_router_rows_exact.py tests/kernels/routing

# 2. the pure-Python window logic
pytest tests/generator/test_cost_aware_window.py

# 3. depth sweep with the deployment settings; compare 7 against 8 and 2 with the usual harness,
#    then re-run the 0.25 quality gate on the winner
EXL3_ROUTER_DET=0 EXL3_DRAFT_TOKENS=7 <harness>
EXL3_ROUTER_DET=0 EXL3_DRAFT_TOKENS=8 <harness>
EXL3_ROUTER_DET=0 EXL3_DRAFT_TOKENS=2 <harness>

# 4. dynamic window: first the calibrated cut alone, then cost-aware with the measured per-row round costs
EXL3_ROUTER_DET=0 EXL3_DYNAMIC_DRAFT=1 <harness>
EXL3_ROUTER_DET=0 EXL3_DYNAMIC_DRAFT=1 EXL3_DRAFT_COST="1:<ms>,3:<ms>,5:<ms>,9:<ms>,16:<ms>" <harness>

# 5. long context: at 16k and 32k, split target 4 and 8 against the default 2
EXL3_ATTN_SPLIT_TARGET=4 <harness>
EXL3_ATTN_SPLIT_TARGET=8 <harness>
```

Fill `EXL3_DRAFT_COST` from your own measured whole-round times by verify rows (window + 1);
the example values in `doc/env_vars.md` are from one shallow-context run and are not a default.
Every comparison uses the six per-request phases and the strict `>5% AND >3 ms` rule, never an
average. Run the AR fingerprint control once with `EXL3_ROUTER_DET=0` on the new binary before
the sweep: the router change moves only multi-row calls, so single-row AR must be identical.

## 3. Interplay with the upstream sync

The upstream `draft_max_rows` ceiling (Generator argument, default 0 = off) and the local
`EXL3_DRAFT_COST` cut both shorten the verify window and both now exist. With the ceiling off
nothing changes; if both are enabled, the ceiling bounds the window first and the cost-aware cut
chooses within it. Measure one at a time before combining them. The upstream batched readback
(`288d0434`) applies to every spec round regardless of these settings.

## 4. Still to do

- Wire `moe_route_sort.py` into the all-fused fast path of `block_sparse_mlp.py` (the tier plan
  around the `expert_count_list is None` branch) behind `EXL3_MOE_FAST_SORT`, with a parity test
  against the torch path. Only worth it if the best depth is above 7 (9..16-row windows).
- The wide head (m = 1 route 90 vs m >= 2 route 92) remains the one dense family that differs
  between AR and any verify window; unchanged by either branch.
