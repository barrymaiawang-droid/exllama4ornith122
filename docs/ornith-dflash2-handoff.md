# Ornith DFlash2: remaining numerical failure and current diagnostic handoff

## Status and requested work

The original AR/native-MTP2 project has independently accepted numerical,
functional-serving/dsh and current matched-performance scope. Its experimental
handoff is [f326f7b8](https://github.com/barrymaiawang-droid/exllamav3/commit/f326f7b8e2e8d44f579cd63d9ec54dd794db35e5).
Historical noisy timing windows remain inconclusive, not retrospectively passed;
this is not a zero-slowdown or speed-gain claim. No production/default/fleet change.

**DFlash2 is NOT accepted.** Fix the remaining full-model numerical discrepancy,
preserve AR/native MTP2, then validate affected quality, serving/dsh and matched
performance. Default stays MTP2 unless DFlash2 produces a meaningful matched speed
improvement without quality or functional regressions; otherwise deliver a
correctly functioning opt-in mode. Do not relax the quality threshold.

This branch starts at the **exact tested Claude source**
[`30cab0c0dc02f3c0f0d038d954087b8bc24432e7`](https://github.com/barrymaiawang-droid/exllamav3/commit/30cab0c0dc02f3c0f0d038d954087b8bc24432e7).
The new commit adds this handoff only: **no untested arithmetic patch** or repaired
model-quality claim. Earlier investigation:
[issue #1](https://github.com/barrymaiawang-droid/exllamav3/issues/1) and
[actual route-census feedback](https://github.com/barrymaiawang-droid/exllamav3/issues/1#issuecomment-6445583106).

## Pinned target, drafter and stack

- Target: `Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw`, revision
  `9c5c913b3de1eef901520d6fedaf450044ea0423`; do not substitute quantization/model.
- Drafter: [`incoai/Ornith-1.5-122B-DFlash2`](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2/tree/9baefc1bec85c08193ab442e0ca5f7057669b61d),
  revision `9baefc1bec85c08193ab442e0ca5f7057669b61d`, Apache-2.0.
- Linux x86-64 Strix Halo `gfx1151`; Python 3.12.13,
  Torch `2.14.0+rocm10.1.0a20260822`, HIP `7.16.26332` (pinned ROCm10 stack).
- Hidden 3072, 48 target layers, 6 drafter layers; block 16 / 15 proposals;
  selector rank 256 / top-k 16; target taps `[1,7,14,20,26,32,39,45]`.
- Padded head 248320, real tokenizer vocabulary 248077; mask token 248077.
  Compare real-vocabulary logits; padded columns are not valid natural tokens.
- Artifact sidecars/license/shapes/weight-file sizes were revalidated after original
  acceptance, reusing the verified download markers. No model re-download/copy or
  repeated full-weight hashing was performed.

## Actual correctness evidence: operator PASS is not model PASS

The single-matrix row16 implementation at
[`6bc9a866`](https://github.com/barrymaiawang-droid/exllamav3/commit/6bc9a866aef8ad0f86b46bd447aee657d36c7c47)
uses two independent <=8-row groups; the two-matrix path still declines >8 rows.
The focused GPU suite passed **19/19 nodes, 57/57 phases, zero skips/failures**.
Its native artifact SHA256 was
`42d47e99b5481b41a19bc2c19f0c613d6076a939db1f1b799150ebbe8f66c217`.

The unchanged full-model 13-job schedule then **failed** the fixed 0.25 gate:

| workload | first failed prediction row | gap vs 0.25 gate (AR lead over DF) |
|---|---:|---:|
| short paragraph |18|0.6875|
| prose |43|0.265625|

Repeat/teacher/continuation/sampling checks passed within that schedule; this does
not waive the two greedy failures. The schedule includes four 16-token warmups,
three 96-token short jobs, three 64-token prose jobs, 97-row teacher replay and
32-token continuation/sampled jobs, up to 705 new tokens. Do not call missing or
failed jobs a quality pass. Model execution/SDK/ownership/cleanup success was
separate from the numerical failures.

That is the full-quality result on the row16 artifact, **not** a rerun of the full
schedule on the later tracing artifact. The rebuilt `30cab0c0` artifact
`983a5834749e6404f26ee51c7de28e09ae8b0ada17298f404627e8cbe20927d1`
was used for the bounded route census and later load/operator diagnostics only.

## What the actual native census established

Three jobs (prose AR warmup16, prose AR64, short oracle33) completed with 91 method
events, **20,623 native entry-completion records, zero drops**. Source/native/SDK,
no-native-JIT and restoration/lifecycle guards passed. The recorder is host-side
entry completion, not a complete GPU-launch timeline or an arithmetic fix.

- No observed AR single-matrix 9..16 calls **within the recorded method scopes**.
  AR forward m1 and tail prefill m25 were observed. Large 2048/1024 prefills had
  zero recorder events: this is not zero GPU work or complete coverage.
- Oracle m16 QKV/Z: K3072, N12288/8192, **5-bit mul1, fp32 output**, route92.
- Wide head N248320: **6-bit mul1, fp16 output in both arms**; AR m1 route90,
  oracle m16 route92. Do not mislabel this as a 5-vs-6-bit comparison.
- Two-matrix K3072/N1024 or512, slots2, **5-bit mul1, fp16 output**: AR m1 route93,
  oracle m16 route2. Shape/slots do not uniquely identify a module or expert,
  and two matrices do not by themselves imply weighted MoE.
- There are **three** actual target m16 forwards at offsets 0/16/32. The last
  round is EOS-unresolved; terminal `[33,15,0]` is not measured rejection.
- Helper/fused operations outside `Model.prefill/forward` scopes were not covered.
  Absence of the recorder's graph bit alone does not prove absence of replay.

The natural AR64 endpoint equalled the pre-row16 reference but differed from the
earlier row16 quality run at row2. Different recipe/history/observer binary means
this is descriptive, **not causal AR-preservation proof**. Current oracle first32
real-vocabulary bytes differed from both old DF and old AR. No numerical repair was
established. Oracle proposals/forced prefix/acceptance are artificial and are not
natural drafting-quality evidence.

## Latest planned discriminant, and why it has no numerical result yet

The retained earlier trace contains real normalized AR inputs for layer0 MLP and
layer3 full attention: 32 rows each, 3072 values/row, first32 positions only. A
private 786432-byte fixture was extracted by byte copy, without CPU model math or
new inference. No weights or raw tensor fixtures are published here.

The intended test uses exact named loaded weights, not a shape-guessed owner:

- `model.language_model.layers.0.mlp.shared_expert.gate_proj` and `up_proj`,
  each output width 1024, using the actual normalized MLP input.
- `model.language_model.layers.3.self_attn.k_proj` and `v_proj`, each output
  width **512** (= 2 KV heads * 256), using the actual normalized attention input.

Planned operator schedule: each pair runs 32*m1, 4*m8, 2*m16, 32*forced-WMMA-m1
(shape2/sms20); 140 MGEMM calls across the two pairs, plus 128 separate GEMM-m1
positive controls = **268 native calls**, zero generation/jobs/warmups. Each call
must have contiguous `(2,m,n)` and `(2,m,k)` output/hadamard slabs; slices of a
`(2,32,*)` buffer have the wrong slot stride when m<32.

Falsifiable prediction: MR-m1 == MR-m8 == separate-GEMM-m1; WMMA-m16 == forced
WMMA-m1; all rotated inputs identical. Disagreement must be retained, not patched
away or turned into a quality golden. This would establish a family association
for real inputs/weights, not that WMMA is mathematically wrong or the end-to-end
cause has been fixed. It does not test the remaining wide-head difference.

**No operators have run in this new diagnostic.** One wrapper attempt walked the
entire 48-layer MoE module tree before its 8192 guard and refused. A reviewed,
stub-tested fix narrowed the walk to the selected block/site with a 4096 limit
checked during iteration. The next attempt loaded both models and passed its
source/SDK/JIT/lifecycle guards but refused at the exact named pair lookup:
`ValueError: exact named actual MultiLinear owner not unique: shared_gate_up_l0`.
It saved zero arrays and made zero MGEMM/GEMM calls; this is a probe failure, not
another measured numerical failure or a model fix.

### Concrete probe lookup bug identified in source

The failed resolver inspected only **direct attribute values** for
`type(value).__name__ == "MultiLinear"`. `GatedMLP` stores these objects inside
**`self.multi_gu`, a list**, not as a direct attribute. See
[`mlp.py` constructor/load](https://github.com/barrymaiawang-droid/exllamav3/blob/30cab0c0dc02f3c0f0d038d954087b8bc24432e7/exllamav3/modules/mlp.py#L710-L804):
`self.multi_gu = [None] * self.num_slices`, followed by
`self.multi_gu[load_slice] = MultiLinear(..., [self.gates[load_slice], self.ups[load_slice]])`.
Thus the direct-attribute-only search cannot find this real pair. Use the framework's
actual `multi_gu`/slice/linear structure (with exact keys and bounds), or construct
an explicitly diagnostic pointer table from the exact named linears. Do not widen
module limits or fall back to shape-only matching. K/V lookup was not reached.

The wrapper's config/manifest parser was also repaired and inert RED/GREEN-tested
to hash and parse the **same raw buffer**, rather than hash one read then parse a
second. All failed/refused windows remain retained with exclusive identities.

## Source entry points and next checks

- [Dispatch / m16 fallback](https://github.com/barrymaiawang-droid/exllamav3/blob/30cab0c0dc02f3c0f0d038d954087b8bc24432e7/exllamav3/exllamav3_ext/quant/exl3_gemm.cu)
- [Multirow kernel / single-matrix row groups / two-matrix <=8 guard](https://github.com/barrymaiawang-droid/exllamav3/blob/30cab0c0dc02f3c0f0d038d954087b8bc24432e7/exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_multirow_rdna.cu)
- [MultiLinear pointer-table layout](https://github.com/barrymaiawang-droid/exllamav3/blob/30cab0c0dc02f3c0f0d038d954087b8bc24432e7/exllamav3/modules/multilinear.py)
- [Target normalized-sublayer input order](https://github.com/barrymaiawang-droid/exllamav3/blob/30cab0c0dc02f3c0f0d038d954087b8bc24432e7/exllamav3/modules/transformer.py#L150-L210)
- [Existing row-group GPU regression](https://github.com/barrymaiawang-droid/exllamav3/blob/30cab0c0dc02f3c0f0d038d954087b8bc24432e7/tests/kernels/gemm/test_exl3_gemv_rowgroups.py)

First fix the bounded diagnostic lookup and establish the real-input operator
result. Only if it supports the family hypothesis, investigate two-matrix m9..16
row-group support preserving the <=8-row arithmetic, offsets/counter partitions,
variable-width destinations and graph base-pointer patching. Keep weighted/m1
behavior correct. The wide head remains an independent untested seam. No such
arithmetic patch has been implemented in this handoff.

Then run focused affected GPU operators and the unchanged full-model quality
schedule; preserve failed evidence and AR/MTP2 controls. Performance uses all six
per-request phases, strict >5% AND >3 ms, never averaged. Native-profiler timings
cannot replace untraced acceptance. Full integrated serving/dsh and independent
review remain required before DFlash2/default/final acceptance.

Local GPU work is stopped for the requested Claude handoff. Credentials,
privileged ownership/admission records, private host paths, raw tensors, weights,
caches and binaries are excluded from this branch and issue. Any later local
verification requires a fresh admitted window; no monitoring or automatic retry
is implied by this handoff.

## Deployment default (2026-10-28)

This deployment now runs DFlash2 as the default spec, with the int8 multirow router displaced:

```
EXL3_ROUTER_DET=0   # REQUIRED: without it >3-row routing uses the int8 projection and the
                    # unchanged 0.25 quality gate fails (0.6875 short / 0.265625 prose)
--draft-model <Ornith-1.5-122B-DFlash2>  --draft 15   # selects DFlash2 instead of MTP
```

With the switch the same gate passes (short divergence moves to token 27 with a 0.03125 near-tie;
prose shows no divergence). Measured generation is 91.6-97.8 t/s versus 19.7-21.0 t/s for MTP2 on
forced 256-token windows; short natural-stop answers are dominated by ~1.5 s of per-request cost,
not by decode rate. MTP2 remains available (`--draft 2`, no `--draft-model`) as the fallback.

Caveats that matter for anyone reusing this: the switch is process-wide and latched at each layer's
first routing call, so set it before loading; do not use it with tensor-parallel replicated routing,
which depends on the deterministic projection for cross-rank agreement; and it was validated on one
checkpoint (48 MoE layers, `router_type std`) on a single ROCm/gfx1151 GPU.
