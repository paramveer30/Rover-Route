"""The rover loop: look, update the cost map, (re)plan, take one step."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .costmap import CostConfig, cost_grid
from .perception import Belief, observe
from .planner import plan
from .world import World

REPLAN_EVERY = 4   # steps; also replans at once if the route ahead becomes blocked


@dataclass
class Frame:
    cell: tuple[int, int]
    path: list[tuple[int, int]]
    seen: np.ndarray
    known: np.ndarray


@dataclass
class Mission:
    mode: str
    trajectory: list[tuple[int, int]]
    reached: bool
    stop_reason: str
    replans: int
    first_path: list[tuple[int, int]]
    belief: Belief
    frames: list[Frame] = field(default_factory=list)


def run_mission(world: World, mode: str, cfg: CostConfig = CostConfig(),
                max_steps: int = 400, record: bool = False) -> Mission:
    belief = Belief.empty()
    cell, path, replans, first_path = world.start, [], -1, []
    trajectory, frames = [cell], []
    use_semantics = mode != "geometry"
    reason = "step limit"
    for step in range(max_steps):
        if cell == world.goal:
            reason = "reached goal"
            break
        observe(world, belief, cell, mode)
        cost, blocked = cost_grid(world, belief, use_semantics, cfg)
        blocked[cell] = False      # the rover is already standing here
        ahead_blocked = any(blocked[c] for c in path[1:])
        if not path or ahead_blocked or step % REPLAN_EVERY == 0:
            new_path = plan(cost, blocked, cell, world.goal)
            if new_path is None:
                reason = "no route to goal"
                break
            if new_path != path:
                replans += 1
            path = new_path
            first_path = first_path or list(path)
        if record:
            frames.append(Frame(cell, list(path), belief.seen.copy(), belief.known.copy()))
        path = path[1:]
        cell = path[0]
        trajectory.append(cell)
    if record:
        frames.append(Frame(cell, list(path), belief.seen.copy(), belief.known.copy()))
    return Mission(mode, trajectory, cell == world.goal, reason, max(replans, 0),
                   first_path, belief, frames)
