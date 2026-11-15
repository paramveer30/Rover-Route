"""A seeded 60 x 60 grid world stitched from 3 x 3 real AI4Mars tiles.

Grid cells are (row, col), row 0 is north. Each cell is 0.5 m, so the map is 30 m wide.
The rover never reads `truth`; only the hazard scoring and the oracle mode do.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .data import ROCK, SAND, TILE, load_tiles

TILES_PER_SIDE = 3
SIZE = TILE * TILES_PER_SIDE      # 60 cells
CELL_M = 0.5
MAX_SLOPE_DEG = 20.0
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

Cell = tuple[int, int]


@dataclass
class World:
    seed: int
    truth: np.ndarray        # (60, 60) uint8 true class, hidden from the rover
    probs: np.ndarray        # (60, 60, 4) model prediction for every cell
    slope: np.ndarray        # (60, 60) degrees, sensed geometrically
    tile_ids: list[list[str]]
    start: Cell
    goal: Cell

    def tile_at(self, cell: Cell) -> str:
        return self.tile_ids[cell[0] // TILE][cell[1] // TILE]


def hazard_mask(world: World) -> np.ndarray:
    """True terrain the rover must not drive on: sand, big rock, or too steep."""
    return (world.truth == SAND) | (world.truth == ROCK) | (world.slope > MAX_SLOPE_DEG)


def _slope_field(rng: np.random.Generator) -> np.ndarray:
    rows, cols = np.mgrid[0:SIZE, 0:SIZE] * CELL_M
    elevation = np.zeros((SIZE, SIZE))
    for _ in range(rng.integers(3, 6)):
        r0, c0 = rng.uniform(0, SIZE * CELL_M, 2)
        height, sigma = rng.uniform(1.0, 2.5), rng.uniform(1.5, 3.5)
        elevation += height * np.exp(-((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * sigma ** 2))
    d_row, d_col = np.gradient(elevation, CELL_M)
    return np.degrees(np.arctan(np.hypot(d_row, d_col)))


def _nearest_safe(safe: np.ndarray, corner: Cell) -> Cell | None:
    cells = np.argwhere(safe)
    if len(cells) == 0:
        return None
    best = cells[np.argmin(np.abs(cells - np.array(corner)).sum(axis=1))]
    return int(best[0]), int(best[1])


def reachable(safe: np.ndarray, start: Cell, goal: Cell) -> bool:
    """8-connected flood fill over safe cells, no cutting diagonal corners."""
    seen = np.zeros_like(safe, dtype=bool)
    seen[start] = True
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < SIZE and 0 <= nc < SIZE) or seen[nr, nc] or not safe[nr, nc]:
                continue
            if dr and dc and not (safe[r + dr, c] and safe[r, c + dc]):
                continue
            seen[nr, nc] = True
            queue.append((nr, nc))
    return False


def build_world(seed: int, max_attempts: int = 100) -> World:
    """Deterministic world for `seed` whose goal is reachable without touching a hazard."""
    tiles = load_tiles()
    for attempt in range(max_attempts):
        rng = np.random.default_rng([seed, attempt])
        pick = rng.choice(len(tiles.ids), TILES_PER_SIDE ** 2, replace=False)
        truth = np.zeros((SIZE, SIZE), np.uint8)
        probs = np.zeros((SIZE, SIZE, 4), np.float32)
        ids = [[""] * TILES_PER_SIDE for _ in range(TILES_PER_SIDE)]
        for k, index in enumerate(pick):
            tr, tc = divmod(k, TILES_PER_SIDE)
            block = np.s_[tr * TILE:(tr + 1) * TILE, tc * TILE:(tc + 1) * TILE]
            truth[block] = tiles.labels[index]
            probs[block] = tiles.probs[index]
            ids[tr][tc] = tiles.ids[index]
        world = World(seed, truth, probs, _slope_field(rng), ids, (0, 0), (0, 0))
        safe = ~hazard_mask(world)
        start = _nearest_safe(safe, (SIZE - 2, 1))
        goal = _nearest_safe(safe, (1, SIZE - 2))
        if start and goal and reachable(safe, start, goal):
            world.start, world.goal = start, goal
            return world
    raise RuntimeError(f"no reachable world for seed {seed}")
