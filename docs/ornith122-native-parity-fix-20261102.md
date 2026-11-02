# Ornith-1.5-122B native parity investigation

The release candidate matched all 16,452 visible tokens across two full eight-row
DFlash runs against their saved AR references. Both comparisons used context capacity
262,144, a 32,768-token output reservation and 2,048-token prefill chunks. A bounded
sampling check and live Node B serving checks also passed. The committed source and
pinned extension were installed and passed the normal-launch smoke; the managed service
was stopped afterward to restore its initial state. The 262,144-token cache was allocated,
but the longest tested prompt was 30,478 tokens and the longest complete sequence about
38,300 tokens; this is not a full-context quality test. These two fixtures do not establish
model-wide bitwise or stochastic-distribution equivalence.

## Release evidence

The deployed engine source is commit `32bad02af5c9f7fdf32555efa40564247ed555a8`,
based on `806a103c2d4cd1d5935c192208b4031ff67faac5` with the changes below. Its
release native extension SHA-256 is
`dd490fab5b42448aafc6b2d4390b7c45d921f034c1d4f2a2e2153e3f6a460bf7`.

| Context | Visible tokens matched | Prefill seconds (tokens/s) | Generated including EOS | Decode seconds (tokens/s) | Accepted drafts | Rejected drafts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First write | 8,630 | 40.09 (557.83) | 8,631 | 259.58 (33.25) | 6,800 | 6,017 |
| Final write | 7,822 | 51.71 (589.41) | 7,823 | 211.15 (37.05) | 6,405 | 3,521 |

Neither run had repetition-guard interventions. These are individual runs against saved
AR references, not an uncontended paired benchmark. Recent shorter AR controls measured
about 23 tokens/s; the full-run prefill figures above are measurements, not a claim of
prefill improvement. Head output checks were bit-identical at rows 1–8 at captured
positions 0, 17 and 2264 in the first case and 0, 4241 and 7597 in the second. An
observer confirmed the original 32-row layout for 13-row short-prefill calls across all
12 attention layers and the 16-page bucket in decode.

A focused sampling check at temperature 0.6, top-k 20, top-p 0.95 and seed 123 also
passed on both contexts. Two natural 128-token runs per mode matched across AR and
DFlash; DFlash rejected 120 proposals per run in the first context and 32 per run in
the second. Forced-prefix score comparisons covered 32 positions per context: all 64
raw-logit rows were bit-identical, with maximum total variation 0 and no finite-mask
differences. This is evidence for these inputs and settings, not arbitrary prompts,
seeds or sampling configurations.

A previous pre-head candidate's 128-position target-only oracle found no greedy argmax
mismatches at verification rows 2, 3, 4 and 8; its AR repeat was bit-identical, while
some verifier logits differed from AR. Those results describe that earlier candidate,
not a universal score guarantee for this release. Known-proposal target-only tests also
do not exercise rejected-draft replay; the full DFlash runs above did exercise rejection.

The [sanitized aggregate evidence](ornith122-dflash-parity-evidence.json) contains the
greedy and sampled check results without prompts or generated token IDs. A live candidate
smoke on Node B port 18161 reported context 262144, DFlash2 depth 15, chunk 2048 and FP32
state; the process exposed the eight-row cap, FP16 router setting and head-parity flag,
and its mapped extension matched the release hash. Both non-streaming and SSE arithmetic
requests returned 703 and completed normally. A second smoke using the installed source,
launcher and pinned extension passed the same non-streaming and SSE checks. After
validation the managed service was stopped to restore its initial state.

Use fleet item 22 to start DFlash2; select AR as the fallback when needed. The rollback
snapshot is `$ROOT/rollbacks/20261102-native-parity`: stop the managed service, verify
the snapshot manifest, restore the saved launcher, spec, fleet entry and note, and leave
the service stopped unless a restart is requested. Preserve unrelated fleet changes.

## Changes and limits

- **MoE scratch sizing.** With top-8 routing, hidden scratch held 64 rows. RDNA decode
  needs two scratch rows per routed slot: four input rows need 64, while five through
  eight need more. The eligibility check sent 5–8-row calls to the cooperative fallback.
  The allocation now accommodates two rows per slot, adding about 384 KiB of shared
  scratch for the tested hidden width. Route traces and synthetic and captured-input
  replays verified RDNA dispatch and identical per-row results at widths 1–8.
- **ROCm group-16 attention.** The split cap previously changed with query width: in
  the observed shape it went from 20 to 10. The candidate preserves the single-query
  cap and assigns each query its own row block, so non-sliding causal attention stops at
  that query's causal end. This reduces K/V reuse across queries; its isolated cost has
  not been measured in a matched before-and-after test. Unmarked short-prefill calls
  retain the original layout. Sliding-window and other geometries are outside this
  specific change.
- **Generator page-table bounds.** The generator pads block tables to 16-page
  increments, and draft lookahead can widen the speculative table before the AR table.
  The candidate passes an explicit generator marker to BC attention and derives each
  query's split bound from its AR-padded causal length. This applies only to marked,
  single-sequence ROCm group-16 causal non-window attention when the host split count is
  saturated throughout the minimum bucket. The generator also trims extra draft
  reservation during prefill to align the target table with the corresponding AR budget.
- **Wide head.** `EXL3_HEAD_AR_PARITY=1` enables the tested one-warp reduction order
  for the matching non-graph ROCm head shape and tiles core; the library default is off
  and the Node B launcher template opts in. In a focused earlier operator experiment,
  a batch took 2.76 ms versus 2.74 ms with the experiment off and about 21 ms for serial
  single-row execution. This does not establish a general performance guarantee.

The scoped attention checks passed 19 CPU and 60 GPU tests; the prefill reservation
change passed two focused CPU tests and the 57-test generator suite. Earlier apparent
divergences against saved output were confounded by different output reservations:
prefill uses the reserved block-table width, which can change reduction order. Match
context capacity, reservation and prefill chunk size when comparing paths.

The original retained extension and a clean build from the source base had matching
native source and normalized flags. The clean control extension SHA-256 was
`4ef0b190aa791680852a8057b4b3c32a08627d104381bc7c2c8164cd2383c04b`; AR and
three-row verification at 32 positions were bit-identical across those binaries. This
is evidence for the tested operations, not proof of binary equivalence.

Prefill remains at the tested 2,048-token chunk size. An earlier 4,096-token run was
faster but changed greedy preferences in a long-context control, including a 1.03125
logit-gap case; it is not adopted as a quality-equivalent default.

## Focused checks

Use the target environment and a matching native extension; set `MODEL` to the local
Ornith checkpoint directory. The MoE diagnostic loads one layer and uses synthetic
inputs, without private prompt fixtures:

```sh
python -u serving/ornith122/diagnostics/dflash_parity/moe_row_parity.py \
  --model "$MODEL" --output /tmp/ornith-moe-row-parity.json
```

The target-only oracle needs a separately retained private fixture directory. Its
`--widths` values are proposal counts, so `1,2,3,7` means verification rows 2, 3, 4, 8:

```sh
python -u serving/ornith122/diagnostics/dflash_parity/isolate_verifier.py \
  --fixtures "$FIXTURES" --model "$MODEL" --output "$RESULTS" \
  --widths 1,2,3,7 --new-tokens 128 --assert-greedy-parity
```

Choose new output paths; do not commit prompts, completions, scores, model weights or
caches. Run GPU checks only when the device and memory are available. The earlier
[handoff](ornith122-dflash-parity-handoff-20261102.md) preserves the original evidence
and the now-falsified eight-row hypothesis as history.
