# Ornith122 DFlash token divergence — Claude Code handoff

Prepared 2 November 2026, Australia/Sydney (AEDT, UTC+11).
Latest numerical experiment completed at 12:50 AEDT. This is an unresolved
diagnostic handoff, not a numerical fix or a declaration that DFlash is accepted.

## Start here

The user wants llmfleet item 22 (Ornith-1.5-122B) to deliver the model's best
quality while improving decode and prefill speed. The immediate task for Claude
is to isolate and fix, or rigorously explain, the target-token discrepancy between
ordinary autoregressive decoding (AR) and speculative block verification.

**Confirmed:** the deployed native engine produces different target scores at
identical token prefixes, and these differences can reverse greedy token choice.
The newest reproduction needs only the target model: no DFlash drafter is loaded
and no draft is rejected. A 16-row target verification block flips zero-based
position 17 from token 8042 (`" refer"`) to 5604 (`" reference"`). Ordinary AR
repeats bit for bit. A 2-row block matches all 32 tested top-token choices, but
its scores are not bit-identical.

**Not established:** a specific faulty kernel/operation, the relative fidelity
of either path to an independent high-precision reference, or systematic average
quality loss. Block arithmetic and full-accept recurrent-state handling are still
candidates. Do not declare this simply a model limitation, or describe speculative
decoding as inherently trading quality for speed.

Use this document for the current findings and deployment defaults. The older
[2026-10-28 DFlash handoff](ornith-dflash2-handoff.md) and
[2026-10-29 performance study](gfx1151-ornith122-baseline-20261029.md) contain useful
dispatch history, but their defaults, historical builds, 0.25 logit-gap gate and
speed claims are not current acceptance evidence. Passing that old tolerance does
not establish exact greedy equivalence. In particular, the old int8-router issue
is not a new explanation: **EXL3_ROUTER_DET=0 was already set in all current tests.**

## Access, ownership and stopping conditions

- Use **Node B only, until 3 November 2026 at 09:00 AEDT** (2 November 22:00 UTC).
  That is the interpreted end of the user's existing allocation, not a recurring
  schedule. Stop diagnostic GPU work by then; if starting after it, obtain a new
  allocation instead of assuming the old one continues.
- **Do not use Node A for inference, benchmarking, builds or diagnostics.** The
  earlier exception allowed Git push only to its fleet Git server. No Node A
  access is needed for this GitHub handoff.
- All model experiments so far used the local Node B host. Its existing launcher
  guard identifies the local interface as `10.55.0.2/24`. Confirm machine identity
  and available memory before loading. The tested gfx1151 host has about 123 GiB
  unified RAM; the 122B target and cache consume substantial memory.
- Do not stop other workloads, change power settings or overwrite unrelated work.
  Another item-2 workload shared Node B during parts of the earlier comparison.
  Serialize large model tests; the target-only test does not need a drafter.
- At the final handoff check, no Ornith service or diagnostic process was running,
  port 18161 was not listening, and the selected profile file contained `ar`.
  Re-check live state before acting. A service being stopped is not a test failure.
- Do not use the local `ornith122-switch.sh`: it contacts Node A. Do not edit the
  user's generated HTML to make the engine test pass. Generated tool calls in all
  diagnostic replays are inert and must remain so.
- No new GPU run is scheduled by this handoff.

## Repository, baseline and deployed files

Continue on, or branch from, `fix/ornith122-serving-quality` in
[exllama4ornith122](https://github.com/barrymaiawang-droid/exllama4ornith122).
This handoff is added on top of the already pushed serving fix:

- Native source used for all current numerical experiments:
  `67681612de24c8772319144b0dd9b7a41177fb22`.
- Serving fix:
  [`9e63e67514d049bbe28ffd1fb323c1aafc8d9c7d`](https://github.com/barrymaiawang-droid/exllama4ornith122/commit/9e63e67514d049bbe28ffd1fb323c1aafc8d9c7d).
  It adds serving/tests/deployment files; native `exllamav3/` is unchanged
  relative to the numerical baseline.
- Fleet fix, already pushed separately:
  `2a35012365f09a91738888b4e9b052d24734b544`, same branch name.
  No new fleet change is part of this handoff.

Node B path map; `HOME` means the existing user's home, not a variable to reassign:

```sh
TASK_ROOT="$HOME/Documents/Codex/2026-11-02/it"
EXP_ROOT="$HOME/x3-exp/b-baseline-delivery-20261026/dflash2-opt-20261027-h1aSJ2/claude-fix-test-v1"
BASE_SOURCE="$EXP_ROOT/source"
NATIVE_ARTIFACT="$EXP_ROOT/artifacts/repguard-build1"
DEPLOYED_RUNTIME="$EXP_ROOT/serving-quality-v5"
FIXTURES="$TASK_ROOT/work/ornith-facing-review"
ENV_SCRIPT="$HOME/x3-exp/b-native-candidate-prep-20261026-kqFr1i/bundle/env-b.sh"
```

The published worktree is `$TASK_ROOT/work/publish-ornith122`. The original native
checkout is still on `guard-periodic` and has a pre-existing untracked
`tools/bench/ornith_http_bench.py`; preserve it. Do not reset/clean either
checkout indiscriminately.

Pinned files and values:

| Item | Value |
| --- | --- |
| Target directory | `$HOME/models/Ornith-1.5-122B-EXL3-EAQv1-4-bpw` |
| Target revision | `9c5c913b3de1eef901520d6fedaf450044ea0423` |
| Drafter directory, only for later integrated checks | `$HOME/models/Ornith-1.5-122B-DFlash2` |
| Extension | `$NATIVE_ARTIFACT/exllamav3_ext.so` |
| Extension SHA256 | `d90691b0eb1ffc1e69fbd565f5c4fa6e2250109df032056eb055165d516f855c` |
| Deployed adapter | `$DEPLOYED_RUNTIME/bundle/ornith_serve.py` |
| Adapter SHA256 | `805d1d38a9775605476c94f373b7aaeecb85bd523948bdd04d6aa1eaa6de2780` |
| Launcher | `$HOME/launch-ornith122-dflash2.sh` |
| Default profile | `$HOME/.ornith122-spec`, currently `ar` |
| Fleet configuration | `$HOME/llmfleet/llmfleet.json` |
| Recorded Torch | `2.14.0+rocm10.1.0a20260822`, one CPU math thread |
| Native cache/context | 262144 tokens, one batch slot, history capacity 15 |
| Current controls | FP32 recurrent state, FP16 router projection (`EXL3_ROUTER_DET=0`), prefill chunks 2048 |

The earlier audit checked all 28 target files against pinned local Hub metadata.
The auxiliary drafter was not independently re-hashed in full. No BF16 reference
model comparison has been done. Keep source and native extension paired: a Python
edit can be tested against the existing artifact only if ABI/behavior permits;
a native edit needs an isolated matching rebuild, with hashes recorded.

## Evidence to read

Published alongside this handoff, without private prompts or large tensors:

- [Verifier isolation measurements](../serving/ornith122/diagnostics/dflash_parity/verifier-isolation.json):
  per-run calls, state settlement, score differences and KL.
- [Full AR/DFlash comparison](../serving/ornith122/diagnostics/dflash_parity/ar-dflash-comparison.json):
  both prompt contexts, geometry checks, prompt hashes and repeat controls.
- [Independent score verification](../serving/ornith122/diagnostics/dflash_parity/score-verification.json):
  common vocabulary, nonfinite checks and the unforced first flip.
- [Runnable target-only isolation](../serving/ornith122/diagnostics/dflash_parity/isolate_verifier.py).

The local narrative reports are under `$TASK_ROOT/outputs/`:
`ornith122-engine-review.txt`, `ornith122-engine-metrics.json`,
`ornith122-facing-check.txt`, `ornith122-ar-dflash-comparison.txt`,
`ornith122-verifier-isolation.txt`, and `ornith122-publish-status.txt`.
Use those rather than repeating the earlier serving audit.

### Strongest current result: target-only, no rejected drafts

The 12:50 experiment loaded one target, no drafter, and ran four cold-cache
conditions on the same 22366-token prompt. Each generated 32 tokens at temperature
0, seed 123, with the same weights, prefill chunk and recurrent precision.
All four completed normally.

The block runs supplied known AR continuation tokens as proposals and forced
those same output tokens. Consequently every proposal matched and no positive
rejection rewind occurred. The score comparison then asked what the target's
unforced argmax would have chosen.

| Condition | Differing argmax positions, zero-based | At position 17: logit(` refer`) minus logit(` reference`) |
| --- | --- | ---: |
| AR baseline | reference | +0.1171875 |
| AR repeat | none; every score bit-identical | +0.1171875 |
| 2-row verification: 1 proposal | none in 32 | +0.1171875 |
| 16-row verification: 15 proposals | 17 | -0.046875 |

The two-row control is **one proposal plus the current input token**. It is not
the earlier serving `NDT=2` profile, which verifies three rows.

For the 16-row run, common-vocabulary mean KL was 0.001896996 and maximum KL
0.036991023. For two rows, mean KL was about 1.57e-7. Raw maximum logit error
alone is not a quality metric: additive shifts cancel under softmax.

This proves a drafter or rejected-draft replay is **not necessary** for the
observed ranking reversal. It does not clear all state handling: the fully
accepted block still calls `rewind(0)`, which settles convolution history and
swaps recurrent-state buffer parity. The first flip is in the second block.
The first block has no argmax mismatches, which is not the same as equal scores.

### Earlier integrated DFlash result

With the actual DFlash2 drafter, the unforced streams share their first 17 tokens,
then AR selects 8042 and DFlash selects 5604. AR favors its token by 0.1171875;
DFlash favors its token by 0.0546875. Thus the original flip is not an artifact of
forcing output, sampler masking, or random sampling.

The 128-position same-prefix comparison found fresh-context argmax mismatches at
17, 32 and 67. All four cold 128-token repeats matched their respective full-run
prefixes. Full sequences first diverged at position 17 for `first_write` and
4241 for `final_write`. The final-context 128-position check cannot explain the
later position-4241 divergence.

The head has 248320 columns; the shared allowed vocabulary is 248077 entries.
Free runs mask 243 padded entries; forced output bypasses that sampler mask.
Compare only common allowed entries. The independently corrected mean KL values
are 0.003794834716 for `first_write` and 0.000528042030 for `final_write`;
use `score_verification` in the published JSON for those corrected numbers.
No NaNs or positive infinities were found in the saved score arrays.

### Why the pelican pages started this investigation

The original request was approximately “make a peligan rides a bike 3d webpage.”
The two initially supplied files were `$HOME/dsh/dsh2/test3/pelican-bike-independent.html`
(DeepSeek Harness) and `$HOME/pelican-bike.html` (OpenCode).
The later facing screenshot corresponds to
`$HOME/dsh/dsh2/test3/pelican-bike-3d.html`, SHA256
`2c3a1481bf383f663eef7e38b2876e39521ad9b2014b46c2aa2784a9f4b49c61`.

That scene has the bicycle and beak along +X, while road/travel run along Z.
An in-memory 90-degree whole-bike rotation fixes alignment. Saved assistant tool
arguments, executed write arguments and disk bytes matched; the serving parser
also preserved the payload. This clears that transport step, not model inference.

Both fresh replays generated correct road alignment. With the earlier faulty
HTML already in conversation history, both AR and DFlash retained the facing
error. The fresh DFlash page also had sideways tire meshes; fresh AR did not.
There was one full sample per mode/context, not a statistical quality evaluation.
The tire-plane check was added after seeing the defect. All four pages completed
and passed JavaScript syntax checks. Geometry was measured with actual Three.js
on CPU and renderer/DOM stubs, not browser visual-quality ratings.

Do not infer that the original facing error proves either a model ceiling or a
DFlash-specific bug. The stronger engine evidence is the same-prefix score flip.
The original page and all original inputs were left unchanged.

## Reproduce the numerical failure

Run only within the Node B allocation after checking available memory/workloads.
The short natural-language prompt alone is not the regression fixture: it is the
reconstructed harness context, including its tools and history.

Required local-only files in `$FIXTURES`:

- `first_write-prompt.txt`
- `prompt-manifest.json`
- `ar-first_write-repeat128.json` (saved reference token IDs; contains private
  completion data, so do not commit it)

The runner validates prompt bytes and encoded token-ID hashes. The first-write
prompt hash is
`812afcc0a55d037d1290f46888050ccf03f6672e27e4387c59b4f1f291f35327`;
its token-ID hash is
`86ecadde64006c1eda88c695aaea0762f596212224cb42e0a09d8c9584a818e3`.
If the files are absent, obtain them from the retained Node B workspace; do not
silently substitute another prompt or publish the private conversation.

From this repository's root, after setting the path-map variables above:

```sh
REPO_ROOT="$(pwd)"
ip -brief address | rg '10.55.0.2/24'
free -h
ss -ltn '( sport = :18161 )'
# Verify Node B and ensure enough free memory before proceeding.
source "$ENV_SCRIPT"
export FORK="$BASE_SOURCE"
export PYTHONPATH="$BASE_SOURCE:$NATIVE_ARTIFACT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 EXL3_ROUTER_DET=0
export PYTHONDONTWRITEBYTECODE=1
unset EXL3_DRAFT_TOKENS
RESULT_PARENT="$(mktemp -d /tmp/ornith122-parity.XXXXXX)"
python -u -P \
  "$REPO_ROOT/serving/ornith122/diagnostics/dflash_parity/isolate_verifier.py" \
  --fixtures "$FIXTURES" \
  --model "$HOME/models/Ornith-1.5-122B-EXL3-EAQv1-4-bpw" \
  --output "$RESULT_PARENT/results" \
  --assert-greedy-parity
```

**Expected on the baseline: exit 1**, after all results are saved, with
`GREEDY_PARITY_FAIL` and `oracle_block16.mismatch_positions == [17]`.
The original diagnostic exited 0 because it recorded a finding rather than
enforcing parity. The published runner adds explicit paths, refuses to overwrite
an existing output directory, and optionally returns a failing status.
Its `run()` and `compare()` function ASTs were checked identical to the executed
version. The CLI adaptation was syntax/help checked; it has not had an additional
full GPU run. The underlying original four-condition run completed successfully.

At the observed baseline each cold condition took about 36–38 seconds after
model load. These times are diagnostic durations, not a speed comparison.
Keep new results outside the repo and preserve the original evidence.

For a candidate, intentionally change `PYTHONPATH` to its isolated source and
matching extension. The baseline command above deliberately imports the pinned
deployed source, not the checkout containing this document. A run that still
imports the baseline is not validation of a patch. Record imported module paths,
Git SHA, extension SHA, flags and model identity for every candidate.

The runner asserts that AR still emits the recorded reference sequence. If a
candidate changes AR, investigate before replacing that reference. The two-row
match is only a 32-position control; do not turn it into a blanket “safe depth”
claim or a production default change.

## Code map and ranked next experiments

Prefer the codebase-memory MCP graph for discovery; index the candidate checkout
first if needed. Native functions below are unchanged at the serving-fix commit.

| Area | Entry points and reason |
| --- | --- |
| Block verification and acceptance | [generator.py](../exllamav3/generator/generator.py): `Generator.iterate_gen` constructs target inputs, sets `recurrent_history`, samples target logits, and continues only while target token equals proposal. `reject_remainder` handles rejected tails. Full acceptance calls `rewind(0)`. |
| State positions and slots | [recurrent_util.py](../exllamav3/cache/recurrent_util.py): `prepare_for_recurrence`, `advance_recurrent_states`; examine base/scratch rows, history length and position. |
| GDN dispatch and settlement | [gated_delta_net.py](../exllamav3/modules/gated_delta_net.py): `GatedDeltaNet.forward`, `_bc_configure_slot`, `GDNState.rewind`, `_rewind_layers`, `GDNLayerState.rewind_conv_job` and `replay_job`. |
| Native recurrent kernels | [gdn.cu](../exllamav3/exllamav3_ext/gdn.cu): convolution settlement and scan replay. No positive rejection replay occurred in the oracle reproduction. |
| Target forward capture | [model_ls.py](../exllamav3/model/model_ls.py): `forward_ls`, and [transformer.py](../exllamav3/modules/transformer.py) for normalized sublayer inputs. |
| Matrix/MoE row-count dispatch | [exl3_gemm.cu](../exllamav3/exllamav3_ext/quant/exl3_gemm.cu), [exl3_gemv_multirow_rdna.cu](../exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_multirow_rdna.cu), [multilinear.py](../exllamav3/modules/multilinear.py), [block_sparse_mlp.py](../exllamav3/modules/block_sparse_mlp.py). The older handoff records useful m1/m8/m16 route observations on other builds. |
| Router | [block_sparse_mlp_routing.py](../exllamav3/modules/block_sparse_mlp_routing.py): `det_router_enabled`. Keep the existing `EXL3_ROUTER_DET=0` baseline; changing it is a separate experiment. |
| Existing focused state test | [test_recurrent_rewind_batched.py](../tests/kernels/recurrent/test_recurrent_rewind_batched.py): `test_gdn_layer_state_replay` compares replay against a prefix scan, not whole-model AR equivalence. |

Suggested order; these are hypotheses and proposed experiments, not measured causes:

1. **Row-count-dependent target arithmetic.** Compare 1, 2, 4, 8, 9 and 16 rows
   using identical target inputs and equal accepted prefixes. Capture the first
   differing layer and normalized input. If a dispatch boundary is responsible,
   the numerical jump should track that boundary. A block-size sweep alone
   does not identify a faulty kernel. Check real named weights/modules against
   a higher-precision/dequantized operator reference; shape-only attribution is
   insufficient.
2. **Full-accept state settlement.** From the same saved state and inputs, compare
   the state after 16 sequential steps with one fully accepted block, including
   convolution tails, recurrent rows, parity, positions and next-step logits.
   If the first significant error appears at settlement, investigate `rewind(0)`
   and scratch/base ownership. Do not assume “no rejection” means “no state code.”
3. **Graph/fusion or precision-path differences.** Once the first suspect module
   is found, disable only its relevant graph/fused path or use its reference
   operation, keeping every other setting fixed. Flags such as
   `EXL3_BC_GDN=0` or `EXL3_GRAPHS=0` are hypotheses, not validated fixes for
   this prompt. Re-check numerical output as well as performance.
4. **Remaining integration/rollback effects.** After target-only parity is
   understood, reintroduce the actual DFlash drafter and deliberately test
   rejection lengths, checkpoint boundaries and consecutive requests. The
   current result rules out rejected-draft replay as a necessary cause of this
   flip; it does not prove that every rejection case is correct.

The older handoff's MultiLinear resolver issue concerns diagnostic lookup through
`GatedMLP.multi_gu`, not a demonstrated generation bug. Avoid repeating that
failed direct-attribute-only lookup. Its historical fixed 0.25 gate and source/
artifact hashes must not be mixed with today's result.

## Validation and completion criteria

First establish a red, inspectable parity check; preserve the current failure
before patching. Reduce prompt/state further if possible without losing it.
A successful fix must explain the mechanism and have a regression test at the
real failure seam, not just alter the expected token or relax a threshold.

Then:

1. Re-run the target-only check with ordinary AR controls and identical-prefix
   logits. Quantify any remaining differences and their ranking effect. A pass
   on one 32-token fixture is necessary evidence, not model-wide proof.
2. Re-run the real DFlash 128-token controls on both `first_write` and
   `final_write`, plus appropriate longer cases, rejection/checkpoint coverage
   and fresh/cached requests. Investigate which path agrees with an independent
   high-precision reference at the first divergent operation.
3. Run focused native tests for changed kernels/state code. Earlier 99 kernel
   tests and 15 guard tests passed, but those results do not clear this failure.
   Preserve serving regressions in [the serving README](../serving/ornith122/README.md):
   15 unit tests, 3000 streaming cases and the HTTP regression harness.
4. Repeat end-to-end generation and geometry/browser checks as needed; do not
   confuse valid JavaScript, correct transport or one good picture with inference
   equivalence.
5. Only then benchmark decode and prefill under matched, uncontended conditions.
   The earlier 11:48 comparison had GPU contention and cannot establish a speed
   ranking. Larger prefill chunks previously changed token choices, so changing
   chunk size is a separate correctness experiment.
6. Remove temporary instrumentation, retain reproducible evidence, record exact
   source/build identities, and commit the actual fix separately from this
   documentation. Keep AR as the deployed default until the evidence supports
   changing it. If ordinary numerical variance explains the result without an
   algorithmic defect, document that with measured operator/reference evidence
   rather than claiming “speculation is supposed to reduce quality.”

## Earlier fixes and interpretation corrections

The serving fix already repaired real prompt/tool transport bugs: flattened
function envelopes, altered tool payload whitespace, lossy history fallback,
reasoning-channel handling, sampling controls, fleet restart environment scope,
disconnect cancellation and timing reporting. See the existing commit and
[deployment review](../serving/ornith122/deploy/node-b/engine-review.txt).
Those findings are separate from the native block-verification discrepancy.
No native arithmetic patch was made in this investigation.

Earlier statements that the facing problem was conclusively “just the model”
were too strong: byte-correct transport does not clear inference. Conversely,
one worse DFlash-generated page does not prove a systematic quality penalty.
The current conclusion is a reproducible native execution difference with an
unresolved cause, narrowed by the target-only oracle experiment.

## Research grounding

Exact speculative decoding preserves the target conditional distribution under
its mathematical/verification assumptions. With greedy decoding and identical
target scores/tie rules, token choices should agree. Under stochastic sampling,
distribution equality does not promise identical text for the same seed.

Primary sources reviewed:

- [Original ICML speculative-decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html):
  exact, distribution-preserving decoding.
- [DFlash paper](https://arxiv.org/abs/2602.06036):
  lossless acceleration is the stated design.
- [DFlash maintainer's explanation](https://github.com/z-lab/dflash/issues/74):
  token-equality verification emits the target sample; the issue reporter's
  contrary claim was rebutted. Do not cite that issue as a confirmed sampling bug.
- [vLLM lossless guarantees](https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding):
  numerical precision and batch-dependent log probabilities can still change
  concrete outputs.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html):
  mathematically equal batched and sliced calculations need not be bit-identical.

These sources explain possible mechanisms; none proves the cause in this ROCm/
EXL3 engine. No third-party reported bug has been established as applicable here.

## Retained local artifacts and suggested skills

All private data remain under `$FIXTURES`. Useful files include
`compare-modes.py`, `run-comparison.sh`, `comparison.log`,
`verify-scores.py`, `score-verification.json`,
`isolate-verifier.py`, `run-isolation.sh`, `verifier-isolation.log`,
`verifier-isolation/*-scores.pt` and the earlier
`*-repeat128-scores.pt` / `*-forced_ar128-scores.pt`.
The original DSH session export is `session.jsonl`; reconstructed request/prompt
files and generated HTML are local-only. Do not publish them, score tensors,
weights, credentials or caches. The HTTP wire body was not archived: fidelity of
the reconstructed prompt is supported by the recorded history, exact conversion
functions and matched token counts, not by a captured wire-byte hash.

Suggested skills for the next agent:

- `diagnosing-bugs` — `$HOME/.agents/skills/diagnosing-bugs/SKILL.md`:
  use the existing red result, minimize it, test competing hypotheses, instrument
  one seam and validate the original reproduction after a fix.
- `tdd` — `$HOME/.agents/skills/tdd/SKILL.md`, if available:
  turn the isolated operation/state failure into a meaningful regression test
  before changing it.
- Use codebase-memory MCP graph discovery per the project's AGENTS instructions.
  Do not create new chats or delegate work merely because this document is a
  handoff; follow the user's current execution instructions.

No unresolved numerical fix is committed here. Claude's next substantive action
is to reproduce the target-only failure on the authorized machine and identify
the first responsible operation/state transition.
