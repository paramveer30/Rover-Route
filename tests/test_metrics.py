import numpy as np

from roverroute.metrics import bootstrap_ci, hazard_entries, safe_shortest_m, score
from roverroute.planner import path_length, plan
from roverroute.world import CELL_M, build_world, hazard_mask


def test_continuous_run_counts_once():
    hz = np.zeros((5, 5), bool)
    hz[0, 1:3] = True
    hz[0, 4] = True
    assert hazard_entries([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)], hz) == 2
    assert hazard_entries([(1, 0), (1, 1)], hz) == 0


def test_success_needs_goal_and_no_hazards():
    world = build_world(0)
    hz = hazard_mask(world)
    safe = plan(np.ones(hz.shape, np.float32), hz, world.start, world.goal)
    result = score(world, safe, reached=True)
    assert result["success"] and result["hazard_entries"] == 0
    assert result["route_ratio"] == 1.0
    assert not score(world, safe, reached=False)["success"]


def test_safe_shortest_matches_planner():
    world = build_world(1)
    hz = hazard_mask(world)
    path = plan(np.ones(hz.shape, np.float32), hz, world.start, world.goal)
    assert abs(safe_shortest_m(world) - path_length(path, CELL_M)) < 1e-9


def test_bootstrap_is_deterministic_and_brackets_mean():
    values = [0, 1, 1, 0, 1, 1, 1, 0]
    low, high = bootstrap_ci(values)
    assert (low, high) == bootstrap_ci(values)
    assert low <= np.mean(values) <= high
