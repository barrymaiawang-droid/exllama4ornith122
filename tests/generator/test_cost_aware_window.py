"""Cost-aware draft cut (generator/draft_confidence.py): pure-Python, no GPU."""

from exllamav3.generator.draft_confidence import VerifyCostModel, cost_aware_window

COST = "1:60.5,3:139.2,5:230.1,9:278.1,16:366.7"   # measured Ornith-122B rounds (rows:ms)


def test_interpolation():
    c = VerifyCostModel(COST)
    assert c(3) == 139.2 and c(16) == 366.7
    assert 230.1 < c(7) < 278.1
    assert c(20) > c(16)


def test_window_bounds_and_monotonic_in_confidence():
    c = VerifyCostModel(COST)
    assert cost_aware_window([], c) == 0
    ws = [cost_aware_window([p] * 15, c) for p in (0.5, 0.8, 0.9, 0.97, 0.995)]
    assert all(1 <= w <= 15 for w in ws)
    assert ws == sorted(ws)
    assert ws[-1] == 15


def test_low_confidence_position_cuts():
    c = VerifyCostModel(COST)
    assert cost_aware_window([0.95, 0.9, 0.9, 0.05] + [0.95] * 11, c) <= 3
