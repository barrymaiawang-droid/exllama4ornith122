from __future__ import annotations
import math


class DraftConfidenceCalibrator:
    """
    Online mapping from a drafter's per-position confidence score to an observed
    acceptance probability, used to truncate draft blocks at the first position whose estimated
    acceptance falls below a target confidence.

    Scores are collected into fixed-width bins of decayed (tested, accepted) counts. Labels come
    only from positions the verifier actually tested: accepted positions, the first mismatch, and
    the bonus token sampled right after a fully accepted (possibly truncated) window, which
    labels the first position below the current threshold.

    The threshold is the lower edge of the lowest bin such that it and every populated bin above
    it show an acceptance rate >= confidence. Sparse bins (effective count < min_count) are
    skipped rather than trusted. Until burn_in labels have been seen, the threshold is -inf
    (full windows).
    """

    def __init__(
        self,
        confidence: float,
        bin_width: float = 1.0,
        decay: float = 0.995,
        min_count: float = 8.0,
        burn_in: float = 64.0,
    ):
        assert 0.0 < confidence < 1.0, "draft_confidence must be in (0, 1)"
        self.confidence = confidence
        self.bin_width = bin_width
        self.decay = decay
        self.min_count = min_count
        self.burn_in = burn_in

        self.bins = {}  # bin index -> [tested, accepted], exponentially decayed
        self.total = 0.0
        self.cached_threshold = None


    def add_label(self, score: float, accepted: bool):
        idx = math.floor(score / self.bin_width)
        b = self.bins.get(idx)
        if b is None:
            b = self.bins[idx] = [0.0, 0.0]
        b[0] += 1.0
        if accepted:
            b[1] += 1.0
        self.total += 1.0
        self.cached_threshold = None


    def decay_step(self):
        """
        Age the statistics; call once per verification round so the mapping tracks drift in
        output style (prose vs code etc.) over a few hundred rounds.
        """
        for b in self.bins.values():
            b[0] *= self.decay
            b[1] *= self.decay
        self.total *= self.decay
        self.cached_threshold = None

    def threshold(self) -> float:
        """
        Confidence score below which draft positions should be cut. -inf while insufficient
        data has been collected; +inf if even the most confident bins fall short of the target
        (recovery then relies on the bonus-token labels of empty windows).
        """
        if self.cached_threshold is not None:
            return self.cached_threshold
        if self.total < self.burn_in:
            return -math.inf  # not cached; keeps returning -inf only until burn-in completes
        thr = math.inf
        any_populated = False
        for idx in sorted(self.bins.keys(), reverse = True):
            tested, accepted = self.bins[idx]
            if tested < self.min_count:
                continue
            any_populated = True
            if accepted / tested >= self.confidence:
                thr = idx * self.bin_width
            else:
                break
        # No bin has accumulated enough labels to judge: still learning, keep full windows so
        # labels keep flowing (cutting everything would starve the statistics instead). +inf is
        # reserved for the case where populated top bins genuinely fail the target
        if not any_populated:
            thr = -math.inf
        self.cached_threshold = thr
        return thr

    def estimate(self, score: float) -> float:
        """
        Estimated conditional acceptance probability for a drafted position with the given
        confidence score, from the nearest populated bin at or below it (falling back to the
        nearest above). Optimistic 1.0 while no statistics are available, so sequential
        drafting keeps producing full windows (and labels) during the learning phase.
        """
        if self.total < self.burn_in or not self.bins:
            return 1.0
        idx = math.floor(score / self.bin_width)
        populated = sorted(k for k, v in self.bins.items() if v[0] >= self.min_count)
        if not populated:
            return 1.0
        below = [k for k in populated if k <= idx]
        k = below[-1] if below else populated[0]
        tested, accepted = self.bins[k]
        return accepted / tested


class VerifyCostModel:
    """
    Per-round cost as a function of verify rows (drafted window + 1), from a measured table such as
    "1:60.5,3:139,5:230,9:278,16:367" (rows:ms, one process, same harness). Linear interpolation between
    points, linear extrapolation past the ends. Used by the cost-aware draft cut: on a wide MoE the verify
    cost grows with the number of distinct experts the window touches, so the window that maximizes
    expected accepted tokens per millisecond is often much shorter than the one a fixed acceptance
    threshold keeps.
    """

    def __init__(self, spec: str):
        pts = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            r, ms = item.split(":")
            pts.append((int(r), float(ms)))
        assert len(pts) >= 2, "EXL3_DRAFT_COST needs at least two rows:ms points"
        pts.sort()
        assert all(ms > 0 for _, ms in pts), "EXL3_DRAFT_COST: costs must be positive"
        self.pts = pts

    def __call__(self, rows: int) -> float:
        pts = self.pts
        if rows <= pts[0][0]:
            (r0, c0), (r1, c1) = pts[0], pts[1]
        elif rows >= pts[-1][0]:
            (r0, c0), (r1, c1) = pts[-2], pts[-1]
        else:
            i = next(i for i in range(1, len(pts)) if pts[i][0] >= rows)
            (r0, c0), (r1, c1) = pts[i - 1], pts[i]
        c = c0 + (c1 - c0) * (rows - r0) / (r1 - r0)
        return max(c, 1e-3)


def cost_aware_window(estimates: list[float], cost: VerifyCostModel) -> int:
    """
    Window w (1..len(estimates)) maximizing expected tokens per round / cost(w + 1), where expected tokens
    is 1 (the bonus / corrected token) plus the sum over i < w of the probability that drafted positions
    0..i are all accepted (estimates are conditional acceptance probabilities per position). The drafter
    has already run when this is decided, so w = 0 (falling back to a plain AR step) is never chosen: the
    table's costs are whole rounds including the drafter, which a skipped verify would still have paid.
    """
    if not estimates:
        return 0
    surv = max(0.0, min(1.0, estimates[0]))
    exp_tokens = 1.0 + surv
    best_w, best_rate = 1, exp_tokens / cost(2)
    for i, p in enumerate(estimates[1:], start = 1):
        surv *= max(0.0, min(1.0, p))
        exp_tokens += surv
        rate = exp_tokens / cost(i + 2)
        if rate > best_rate:
            best_w, best_rate = i + 1, rate
    return best_w
