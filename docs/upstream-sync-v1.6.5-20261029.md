# Upstream sync: selected turboderp-org commits from v1.6.4 → v1.6.5

Date: 2026-10-29. Base: this repo's `master` (`64a8c56`, the rewritten fork history whose content
base is upstream `da68d57d`, v1.6.4). Upstream reviewed up to `cdc785db` (v1.6.5), 44 commits.
Fourteen were taken, each cherry-picked with `-x` so its message ends in
`(cherry picked from commit <upstream sha>)`. Nothing was compiled or run on a GPU in the
cloud session that prepared this; section 3 is what the local agent has to do.

Selection rule: performance or correctness changes that matter to the Ornith-1.5-122B /
DFlash2 / MTP2 deployment on gfx1151, and that do not change kernel arithmetic (every
accepted AR fingerprint in `docs/gfx1151-ornith122-baseline-20261029.md` and the DFlash2
0.25 gate rely on the arithmetic staying put). Trivia, model-specific additions and
anything that moves fp32 reduction order were left out.

## 1. Taken

| upstream | change | why here |
|---|---|---|
| `288d0434` | Generator: batch the speculative-window token readback | one device→host readback per verify round instead of per token; shortens every DFlash2 / MTP2 round |
| `2d51dd4b` | Generator: accept `prefill_cooldown=None` | callers may pass None (serve wrapper) |
| `087f035e` | tests: equivalence of the batched readback | the test that guards `288d0434` |
| `afd64978` | Followup #560: no batched readback with a pending `constrain_output_now` or dormant trigger-token filter | correctness of the above with filters (dsh tool use) |
| `85cae3e7` | Generator: `draft_max_rows`, a row ceiling for the dynamic draft window | shrinks the verify window as the batch grows; same lever as the depth sweep. Default 0 = off |
| `81bd3a52` | Generator: document the measured row ceiling | docstring for the above |
| `51d1e799` | Generator: one extension call stages the per-step attention params (`generator_stage_decode`) | removes a dozen host dispatches per decode step; pure host C++ (`exllamav3_ext/generator/staging.cpp`) |
| `70d6662e`, `9299fd7f` | fixes to that staging (device strings, single device) | required with `51d1e799` |
| `dc832daf` | Sampler: derive `reqs_past_ids` / `reqs_torch_seed` after identity-step simplification | identity steps no longer force a full-context H2D upload every token |
| `f0dd8f6b` | DFlash: `model.load(draft_model=)` budgets the tap-layer hidden states; `BC_GatedMLP` pair buffers become views of the statics | memory correctness for exactly this drafter (`export_state_layers` is set by the fork's DFlash2); fewer per-layer allocations |
| `1407bceb` | Followup #546: `list_tensors(prefix)` strips a trailing "."; applied to DFlash2 | DFlash2 `candidate_selector` tensors were looked up as `candidate_selector..` |
| `b470a67d` | Tokenizer: preserve special-token mode across the chunking probe | the quality fixtures encode with special tokens on |
| `33435f22` | Tokenizer: chunking around double-CRLF tokens | correctness, long prompts |

Native files touched: `exllamav3_ext/bindings.cpp`, `exllamav3_ext/generator/staging.{cpp,h}`
(new), `exllamav3_ext/libtorch/mlp.{cpp,h}`, `exllamav3_ext/libtorch/attention.{cpp,h}`.
**A rebuild is mandatory**: a stale precompiled extension raises
`AttributeError: ... has no attribute 'generator_stage_decode'` on the first decode step.

## 2. Deliberately not taken

| upstream | change | reason |
|---|---|---|
| `d33d1869`, `ac2702b7` | GDN b/a GEMV for up to 16 rows in one pass over the weights | perf, but a different reduction order (8 warps split k, `fdot2` on ROCm) than the current single-row kernel, on AR and verify alike: every GDN layer's beta / gate moves at the rounding level and the retained AR fingerprints stop matching. The projection is small for this model. Take it when re-baselining is acceptable. |
| `ca518f52` | fold a block's residual add into the next block's RMSNorm | same: changes the arithmetic of the residual stream, and touches what the captures mutate in place |
| `47d42fea`, `a6bcdc76` | sparse-head mode: fully on-device MTP drafting loop | 1,450 lines of new modules and tests, untested on ROCm; worth a dedicated port if MTP2 stays the default |
| `56ffdaea`, `f00d7485` | sliced prefill scratch (Turing fa75 + quantized-cache staging) | CUDA Turing kernel plus quantized-cache prefill; not this deployment's path |
| `d4117b7b`, `10617057`, `abdb6331`, `a98ece6f` | CPU-MoE kernels, load fixes | expert offload is not used here |
| `c60b33be`, `4b1af9d4` | convert.py Hessian capture, QBench | conversion / benchmarking only |
| `f8166e1d`, `dfc0802e`, `0233cfc4`, `e7af3420` | NemotronH, Nemotron vision, Gemma4 | other models |
| `1a94d7a7`, `935d4e99`, `cdc785db` | eval script, typo, version bump | trivia |

## 3. For the local agent: how to build, test and judge this

Everything below runs on the pinned box (Python 3.12.13, torch 2.14.0+rocm10.1.0a20260822,
HIP 7.16, gfx1151). Use the same admitted build procedure as for every earlier candidate;
nothing here needs new flags or packages.

### 3.1 Build

```
git fetch origin && git checkout master && git pull --ff-only
# rebuild the extension with the pinned toolchain, the way the accepted binaries were built
# (precompiled extension: rebuild and reinstall it; JIT path: clear the extension cache dir
# so the new generator/staging.cpp is compiled in)
python -c "from exllamav3.ext import exllamav3_ext as e; print(hasattr(e, 'generator_stage_decode'))"
```

The last line must print `True`. Record the new extension's SHA256 as usual.

### 3.2 Tests, cheapest first

```
# no GPU needed
pytest tests/tokenizer/test_chunking.py
pytest tests/kernels/gemm/test_exl3_route_trace.py tests/kernels/gemm/test_exact_linear.py

# GPU, synthetic operands (regressions this tree already carries; none should move)
pytest tests/kernels/gemm tests/kernels/routing tests/kernels/recurrent -x

# GPU + registry model: the batched readback must be token-for-token equal to the old path
pytest tests/generator/test_batch_readback_equivalence.py --model-root <root with tests/models.yaml roles>

# the generator suite around drafting
pytest tests/generator -x
```

`test_batch_readback_equivalence.py` needs the `dense-llama` role from `tests/models.yaml`
(any small dense model registered under that role); it does not need Ornith.

### 3.3 Behavioural checks on Ornith (the ones that decide acceptance)

1. **AR fingerprint control, before anything else.** Run the short (`Write a short paragraph
   about the ocean.`, 96 tokens) and prose (3098-token rendered fixture, 64 tokens) AR jobs with
   `ArgmaxSampler` on the new binary and compare token IDs against the retained AR references.
   Prediction: **identical**. None of the fourteen commits touches a kernel's arithmetic, so a
   difference means a routing or sampler behaviour change and must be bisected over the fourteen
   (most likely `dc832daf` or `288d0434`), not accepted.
2. **Route census unchanged.** `EXL3_ROUTE_TRACE=1` over one AR forward and one 16-row verify
   forward; the table must match the one in the baseline document (same routes per shape).
3. **DFlash2 and MTP2 schedules.** Run the unchanged 13-job quality schedule with the same
   settings as before (`EXL3_ROUTER_DET=0`, the deployed draft depth). The 0.25 gate result is
   expected to be unchanged by this sync; it is reported, not re-interpreted.
4. **Timing.** The readback batching and the staging call cut host work per round. Measure with
   the six per-request phases and the strict `>5% AND >3 ms` rule against the previous binary;
   expect decode rounds to get shorter, prefill unchanged. Do not average.
5. **Serving smoke.** Streaming, tool use with filters (the `afd64978` path), cancellation and
   continuation through the serve wrapper, since the readback and staging changes sit in the
   generator's step loop.

### 3.4 New knobs

- `Generator(draft_max_rows=N)`: row ceiling `bsz * (1 + draft tokens)` per round; 0 (default)
  keeps the old behaviour. For a single stream it only matters if N < 1 + draft tokens.
- `Model.load(..., draft_model=drafter)`: with autosplit, budgets VRAM for the tap-layer states.
  Load the drafter first and pass it. Single-device loads are unaffected.
- `generator_stage_decode` has no switch; `_NO_BATCH_READBACK` in `generator/generator.py`
  is a module flag the equivalence test flips to compare both readback paths.

### 3.5 Interplay with `claude/perf-headroom`

That branch edits the same generator region (`EXL3_DRAFT_TOKENS`, `EXL3_DYNAMIC_DRAFT`,
`EXL3_DRAFT_COST`). Rebasing it onto this master will conflict in `generator/generator.py`
around the draft-window logic, and `draft_max_rows` overlaps in intent with the cost-aware
window. Resolve by keeping both mechanisms side by side (ceiling first, then the cost cut), or
pick one; do not take both blindly.

### 3.6 What was verified before the merge, and what was not

Verified on a CPU host: every changed Python file compiles and passes pyflakes with no new
warnings; all touched modules import with the native extension stubbed; the tokenizer chunking
test passes (16/16); the fork's no-GPU tests pass; the picks applied with no conflicts.
Not verified: the ROCm build of `staging.cpp` and the `mlp.cpp` view change; the readback
equivalence test; any Ornith behaviour or timing. Those are section 3.
