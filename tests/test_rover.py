import numpy as np
import pytest

from roverroute.perception import MODES
from roverroute.rover import run_mission
from roverroute.world import SIZE, build_world, hazard_mask, reachable


def test_world_is_deterministic_and_reachable():
    a, b = build_world(3), build_world(3)
    assert np.array_equal(a.truth, b.truth) and a.start == b.start and a.goal == b.goal
    assert a.truth.shape == (SIZE, SIZE)
    assert reachable(~hazard_mask(a), a.start, a.goal)


@pytest.mark.parametrize("mode", MODES)
def test_mission_runs(mode):
    world = build_world(2)
    mission = run_mission(world, mode, record=True)
    assert mission.trajectory[0] == world.start
    assert mission.frames and mission.stop_reason
    steps = zip(mission.trajectory, mission.trajectory[1:])
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in steps)


def test_geometry_mode_never_learns_terrain():
    mission = run_mission(build_world(2), "geometry")
    assert not mission.belief.known.any() and mission.belief.seen.any()
