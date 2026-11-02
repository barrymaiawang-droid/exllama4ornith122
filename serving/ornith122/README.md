# Ornith-1.5-122B serving adapter

OpenAI-compatible text serving for the EXL3 Ornith checkpoint. This adapter preserves
native tool envelopes and exact string payloads, validates native prompt history,
honors model sampling defaults and fleet controls, separates reasoning from answers,
and cancels abandoned SSE requests during prefill.

The tested Node B candidate uses DFlash2 with an eight-row cap, FP32 recurrent state,
FP16 router projections, one CPU math thread and 2048-token prefill chunks. It uses source revision `806a103c2d4cd1d5935c192208b4031ff67faac5` plus the
patches described below. Its release extension has SHA-256
`dd490fab5b42448aafc6b2d4390b7c45d921f034c1d4f2a2e2153e3f6a460bf7`; it was built
incrementally from the isolated head experiment and GPU validated. Two full DFlash
outputs matched their saved AR references at matched context and output reservations.
Its live candidate smoke passed; the durable source installation remains pending. The
adapter is byte-identical to the reviewed v5 deployment.

The [native parity report](../../docs/ornith122-native-parity-fix-20261102.md) describes
candidate fixes for MoE scratch sizing, ROCm group-16 attention and the wide head, with
their measured scope, bounded sampling check and candidate serving smoke. The earlier hypothesis that a cap of eight
verification rows alone ensures parity was falsified. `VERIFY_ROWS` remains a scheduling
control. The repository template prepares DFlash2 as the default with AR available as a
fallback. The tested live candidate uses DFlash2; the durable installation has not yet
been validated.

## CPU regression checks

Install Jinja2 (see `requirements.txt`), then run from the repository root:

```sh
python -m unittest discover -s serving/ornith122/tests -p 'test_*.py'
python serving/ornith122/tests/test_tool_streaming.py
python serving/ornith122/tests/test_repetition_guard_http.py
```

The 22 unit tests need no GPU or model weights (the seven parity-comparison tests skip
without torch). The 3000 randomized streaming cases read their corpus from the Node B
home directories and only run there.
The real native template is pinned in the fixtures; set `ORNITH_MODEL_DIR` to exercise
another local checkpoint's template. The HTML fixtures check exact file preservation.

## Serving

Use an existing working ExLlamaV3 installation and matching native extension:

```sh
python serving/ornith122/bundle/ornith_serve.py \
  --model /path/to/Ornith-1.5-122B-EXL3-EAQv1-4-bpw \
  --draft 0 --recurrent-state-dtype fp32 --chunk-size 2048 \
  --ctx 262144 --host 127.0.0.1 --port 18161
```

Production requires the matching model, tokenizer and native runtime. The server's
standalone draft argument defaults to MTP2; pass `--draft 0` for ordinary decoding.
The model's generation configuration supplies default temperature/top-k/top-p.

`deploy/node-b/` records the reviewed machine-specific launcher, selected profile and
validation notes. Its absolute runtime/build paths are intentional deployment values;
it is not a portable installer. The launcher honors `PROFILE`, `CTX`, `REASONING`,
`RBUDGET`, `PREFILL_CHUNK`, `RECURRENT_STATE_DTYPE` and `VERIFY_ROWS`. DFlash's `NDT`
override sets what the drafter proposes; `VERIFY_ROWS` bounds what the target verifies
per round. `restart` keeps fleet environment settings in scope.

## Validation and limits

The earlier v5 review passed 99 targeted kernel tests, 15 repetition-guard tests, the adapter
regressions and live HTTP checks in AR, MTP2 and DFlash2. All 28 main-checkpoint files
matched pinned local Hub metadata. The auxiliary DFlash checkpoint was not fully
re-hashed. Node A was not used for model work.

Earlier v5 two-run FP32 probes measured about 23 decode tokens/sec in AR, 31 in MTP2 and 33 in
DFlash2 with NDT=2; AR 8k prefill was about 568 tokens/sec. These are exploratory
measurements under Node B's existing 90W profile and desktop background load.
Those earlier speculative paths changed reference token choices; they predate the parity
candidate. Larger prefill chunks changed the long-context control and remain explicit
options. FP32 state does not establish BF16 equivalence. The candidate allocated a
262144-token cache, but its longest tested prompt was 30478 tokens and its longest tested
complete sequence was about 38300 tokens; this is not a full-context quality test.

The adapter is text-only. Native XML closing delimiters remain reserved. The model
still makes coding/layout mistakes: a complete page passed syntax checks but failed
browser rendering; a targeted model-generated edit repaired it through native tool
history. Rendering then worked, with poor camera framing. This is not a guarantee
of acceptable visual design. Captured private requests and large debug tensors are
not part of this package.

To check the opt-in head reduction against per-row autoregressive calls using
synthetic inputs and only the checkpoint's head weights, run on the target ROCm
system with the intended extension installed:

```bash
EXL3_HEAD_AR_PARITY=1 python serving/ornith122/diagnostics/dflash_parity/head_row_parity.py \
  --model /path/to/Ornith-checkpoint --output /new/head-row-parity.json
```

The report records the extension hash, routes when tracing is available, and
bitwise comparisons for widths 1–8. An unset or zero flag can be used to
measure the original path; mismatches produce a nonzero exit.
