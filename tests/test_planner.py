import numpy as np

from roverroute.planner import path_length, plan


def grid(n=10):
    return np.ones((n, n), np.float32), np.zeros((n, n), bool)


def test_straight_line():
    cost, blocked = grid()
    path = plan(cost, blocked, (0, 0), (0, 9))
    assert path[0] == (0, 0) and path[-1] == (0, 9) and len(path) == 10


def test_goes_around_wall():
    cost, blocked = grid()
    blocked[0:9, 5] = True
    path = plan(cost, blocked, (0, 0), (0, 9))
    assert path is not None and not any(blocked[c] for c in path)
    assert any(r == 9 for r, _ in path)


def test_no_corner_cutting():
    cost, blocked = grid(3)
    blocked[0, 1] = blocked[1, 0] = True
    assert plan(cost, blocked, (0, 0), (1, 1)) is None


def test_prefers_cheap_cells():
    cost, blocked = grid()
    cost[0, 1:9] = 50
    path = plan(cost, blocked, (0, 0), (0, 9))
    assert all(r > 0 for r, c in path[1:-1])


def test_path_length():
    assert path_length([(0, 0), (0, 1), (1, 2)], 0.5) == 0.5 + 0.5 * 2 ** 0.5
