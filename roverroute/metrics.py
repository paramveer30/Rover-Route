"""Score missions against the true terrain. Evaluation only: this reads world.truth."""
from __future__ import annotations

import numpy as np

from .planner import path_length, plan
from .world import CELL_M, World, hazard_mask


def hazard_entries(trajectory: list[tuple[int, int]], hazards: np.ndarray) -> int:
    """Times the rover drove onto hazard cells; a continuous run counts once."""
    entries, inside = 0, False
    for cell in trajectory:
        now = bool(hazards[cell])
        entries += now and not inside
        inside = now
    return entries


def safe_shortest_m(world: World) -> float | None:
    """Shortest route that never touches a true hazard (the best any rover could do)."""
    hazards = hazard_mask(world)
    path = plan(np.ones(hazards.shape, np.float32), hazards, world.start, world.goal)
    return None if path is None else path_length(path, CELL_M)


def score(world: World, trajectory: list[tuple[int, int]], reached: bool) -> dict:
    entries = hazard_entries(trajectory, hazard_mask(world))
    length = path_length(trajectory, CELL_M)
    best = safe_shortest_m(world)
    return {
        "reached": reached,
        "hazard_entries": entries,
        "success": reached and entries == 0,
        "path_m": round(length, 2),
        "route_ratio": round(length / best, 4) if reached and best else None,
    }


def bootstrap_ci(values, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    data = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), (n, len(data)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)
