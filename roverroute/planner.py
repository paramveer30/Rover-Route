"""8-connected A* over a cost grid. Diagonal moves may not cut blocked corners."""
from __future__ import annotations

import heapq
import math

import numpy as np

Cell = tuple[int, int]
SQRT2 = math.sqrt(2)
_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def _octile(a: Cell, b: Cell) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (SQRT2 - 1) * min(dr, dc)


def plan(cost: np.ndarray, blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Cheapest path from start to goal, or None. `cost` must be >= 1 so the heuristic is admissible."""
    rows, cols = cost.shape
    if blocked[goal]:
        return None
    best = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    frontier = [(_octile(start, goal), 0.0, start)]
    while frontier:
        _, g, cell = heapq.heappop(frontier)
        if cell == goal:
            path = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            return path[::-1]
        if g > best.get(cell, math.inf):
            continue
        r, c = cell
        for dr, dc, step in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                continue
            if dr and dc and (blocked[r + dr, c] or blocked[r, c + dc]):
                continue
            ng = g + step * 0.5 * (cost[r, c] + cost[nr, nc])
            if ng < best.get((nr, nc), math.inf):
                best[(nr, nc)] = ng
                came_from[(nr, nc)] = cell
                heapq.heappush(frontier, (ng + _octile((nr, nc), goal), ng, (nr, nc)))
    return None


def path_length(path: list[Cell], cell_m: float) -> float:
    """Length of a path in metres."""
    return sum(math.dist(a, b) for a, b in zip(path, path[1:])) * cell_m
