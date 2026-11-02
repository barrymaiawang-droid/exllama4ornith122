# Ornith-1.5-122B serving adapter

OpenAI-compatible text serving for the EXL3 Ornith checkpoint. This adapter preserves
native tool envelopes and exact string payloads, validates native prompt history,
honors model sampling defaults and fleet controls, separates reasoning from answers,
and cancels abandoned SSE requests during prefill.

The Node B deployment uses AR decoding, FP32 recurrent state, FP16 router projections,
one CPU math thread and 2048-token prefill chunks. Native engine baseline:
`67681612de24c8772319144b0dd9b7a41177fb22`; the native extension was not rebuilt.
The adapter is byte-identical to the reviewed v5 deployment.

The [2 November DFlash parity handoff](../../docs/ornith122-dflash-parity-handoff-20261102.md)
records the native token divergence, a target-only reproduction without rejected
drafts, measured results and the next diagnostic steps. Its later section traces the
divergence to a kernel-tier boundary: verification passes of up to 8 rows stay on the
RDNA tiers whose per-row arithmetic is AR decoding's, while the deployed 16-row DFlash2
passes (`NDT=15`) fall to the cooperative WMMA mgemm and the fused prefill MoE tier.
The launcher's `VERIFY_ROWS` (default 8, `EXL3_DRAFT_MAX_ROWS`) now caps every
speculative pass at that boundary for the `dflash2` and `mtp2` profiles, and
`diagnostics/dflash_parity/isolate_verifier.py --widths` sweeps the row count. AR
remains the selected profile until the sweep and the 128-token controls are rerun on
Node B.

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

Node B checks passed 99 targeted kernel tests, 15 repetition-guard tests, the adapter
regressions and live HTTP checks in AR, MTP2 and DFlash2. All 28 main-checkpoint files
matched pinned local Hub metadata. The auxiliary DFlash checkpoint was not fully
re-hashed. Node A was not used for model work.

Two-run FP32 probes measured about 23 decode tokens/sec in AR, 31 in MTP2 and 33 in
DFlash2 with NDT=2; AR 8k prefill was about 568 tokens/sec. These are exploratory
measurements under Node B's existing 90W profile and desktop background load.
MTP2 and DFlash still changed reference token choices, and larger prefill chunks
changed the long-context control, so they remain explicit options. FP32 state does
not establish BF16 equivalence. The full 262144-token capacity was not re-evaluated.

The adapter is text-only. Native XML closing delimiters remain reserved. The model
still makes coding/layout mistakes: a complete page passed syntax checks but failed
browser rendering; a targeted model-generated edit repaired it through native tool
history. Rendering then worked, with poor camera framing. This is not a guarantee
of acceptable visual design. Captured private requests and large debug tensors are
not part of this package.
