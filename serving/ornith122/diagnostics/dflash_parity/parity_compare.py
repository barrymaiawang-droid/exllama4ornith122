"""Score comparison and provenance capture for the Ornith target-only parity runner.

Pure functions over saved score tensors, importable without a model so the comparison
itself is covered by a CPU test (serving/ornith122/tests/test_dflash_parity_compare.py).
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

import torch

# Flags that change which kernel tier a verification pass takes, or how its operators round.
# Recorded with every run so a result can never be attributed to the wrong build or tier.
TIER_FLAGS = (
    "EXL3_DRAFT_TOKENS", "EXL3_DRAFT_MAX_ROWS", "EXL3_GEMV_MAX_M", "EXL3_MGEMV_MAX_M",
    "EXL3_ROUTER_DET", "EXL3_BC_GDN", "EXL3_GRAPHS", "EXL3_QKV_SLICE", "EXL3_MOE_FUSED_DET",
    "EXL3_MOE_FUSED_ROWS", "EXL3_MOE_MTILE", "EXL3_MOE_BATCH_RECON", "EXL3_ROCM_MOE_DECODE",
    "EXL3_ROCM_HALF_GEMV", "EXL3_GDN_PROJ_FP32", "EXL3_GDN_GATE_FP32", "EXL3_INT8_GEMV",
    "EXL3_DYNAMIC_DRAFT", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH", "FORK",
)


def parse_widths(spec: str) -> list[int]:
    """'15,1' -> [15, 1]: proposal counts per block run, each 1..16, no repeats."""
    widths = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        w = int(part)
        if not 1 <= w <= 16:
            raise ValueError(f"block width {w} outside 1..16 proposals")
        if w in widths:
            raise ValueError(f"block width {w} given twice")
        widths.append(w)
    if not widths:
        raise ValueError("no block widths")
    return widths


def compare(name: str, ref: torch.Tensor, candidate: torch.Tensor, probe=None) -> dict:
    """Greedy-parity comparison of two (positions, vocab) score tensors.

    Only entries finite in both are compared (the forced path bypasses the sampler's padding
    mask). Reports every argmax mismatch, the first one, per-position worst deltas, the AR
    top-two gap at each mismatch (how close the call was) and the KL(ref || candidate) over
    the common vocabulary, mean and max over positions. `probe` = (position, token_a, token_b)
    adds the logit(token_a) - logit(token_b) gap of both runs at that position.
    """
    assert ref.shape == candidate.shape, (ref.shape, candidate.shape)
    ref = ref.float()
    candidate = candidate.float()
    allowed = ref.isfinite() & candidate.isfinite()
    a = ref.masked_fill(~allowed, -torch.inf)
    b = candidate.masked_fill(~allowed, -torch.inf)
    ia, ib = a.argmax(-1), b.argmax(-1)
    mismatches = (ia != ib).nonzero().flatten().tolist()
    deltas = (ref - candidate).masked_fill(~allowed, 0)
    per_pos_max = deltas.abs().amax(-1)
    per_pos_mean = deltas.abs().sum(-1) / allowed.sum(-1).clamp(min=1)

    # KL over the common vocabulary, in fp64 so tiny divergences are not swamped by log-softmax rounding
    la = torch.log_softmax(a.double(), -1)
    lb = torch.log_softmax(b.double(), -1)
    pa = la.exp()
    kl = (pa * (la - lb)).masked_fill(~allowed, 0).sum(-1)

    top2 = a.topk(2, dim=-1).values
    ar_gap = (top2[:, 0] - top2[:, 1])

    result = {
        "name": name,
        "positions": int(ref.shape[0]),
        "mismatch_positions": mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "first_block_mismatches": [p for p in mismatches if p < 16],
        "mismatch_detail": [
            {"position": p, "ref_id": int(ia[p]), "candidate_id": int(ib[p]),
             "ref_top2_gap": float(ar_gap[p]),
             "candidate_gap_ref_over_candidate": float(b[p, ia[p]] - b[p, ib[p]])}
            for p in mismatches
        ],
        "max_logit_abs_delta": float(deltas.abs().max()),
        "mean_logit_abs_delta": float(deltas.abs().sum() / allowed.sum()),
        "per_position_max_abs_delta": [float(x) for x in per_pos_max],
        "per_position_mean_abs_delta": [float(x) for x in per_pos_mean],
        "ref_top2_gap": [float(x) for x in ar_gap],
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "per_position_kl": [float(x) for x in kl],
        "bitwise_equal": bool(torch.equal(ref, candidate)),
    }
    if probe is not None:
        pos, ta, tb = probe
        if pos < ref.shape[0]:
            result["probe"] = {
                "position": pos, "token_a": ta, "token_b": tb,
                "ref_id": int(ia[pos]), "candidate_id": int(ib[pos]),
                "ref_gap_a_over_b": float(a[pos, ta] - a[pos, tb]),
                "candidate_gap_a_over_b": float(b[pos, ta] - b[pos, tb]),
            }
    return result


def summarize(comparisons: list[dict]) -> list[dict]:
    """One row per block run for a sweep table: width, mismatches, first flip, deltas, KL."""
    return [
        {"name": c["name"], "mismatch_positions": c["mismatch_positions"],
         "first_mismatch": c["first_mismatch"],
         "max_logit_abs_delta": c["max_logit_abs_delta"],
         "mean_logit_abs_delta": c["mean_logit_abs_delta"],
         "mean_kl": c["mean_kl"], "max_kl": c["max_kl"],
         "bitwise_equal": c["bitwise_equal"]}
        for c in comparisons
    ]


def _sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_head(path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(extra_modules=()) -> dict:
    """Imported module paths, Git SHAs of their checkouts, extension hash, tier flags, Torch and host.

    Candidate runs are only validation of a patch when `modules.exllamav3` points at the
    candidate checkout and `extension_sha256` at its matching build.
    """
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "argv": list(sys.argv), "flags": {k: os.environ.get(k) for k in TIER_FLAGS}}
    try:
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = None
    mods = {}
    for name in ("exllamav3", "exllamav3.ext", "exllamav3.generator.generator",
                 "exllamav3.modules.gated_delta_net", "exllamav3.modules.block_sparse_mlp") + tuple(extra_modules):
        m = sys.modules.get(name)
        f = getattr(m, "__file__", None) if m is not None else None
        mods[name] = f
    out["modules"] = mods
    pkg = mods.get("exllamav3")
    if pkg:
        out["source_git_head"] = _git_head(os.path.dirname(os.path.dirname(pkg)))
    ext = sys.modules.get("exllamav3.ext")
    so = getattr(getattr(ext, "exllamav3_ext", None), "__file__", None) if ext is not None else None
    out["extension_path"] = so
    out["extension_sha256"] = _sha256(so) if so else None
    return out


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
