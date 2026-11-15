"""Turn the rover's belief into a cost grid (>= 1) and a blocked mask for the planner."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import ROCK, SAND
from .perception import Belief
from .world import MAX_SLOPE_DEG, World


@dataclass(frozen=True)
class CostConfig:
    w_slope: float = 2.0                          # cost for slope, scaled by slope / limit
    w_terrain: float = 4.0                        # cost for terrain risk
    w_unc: float = 3.0                            # cost for model uncertainty
    risk: tuple = (0.2, 0.1, 1.0, 1.0)            # soil, bedrock, sand, big rock
    unknown_unc: float = 0.5                      # uncertainty assumed for unseen cells
    p_block: float = 0.5                          # block sand / big rock at this probability


def cost_grid(world: World, belief: Belief, use_semantics: bool,
              cfg: CostConfig = CostConfig()) -> tuple[np.ndarray, np.ndarray]:
    # Slope is only known where the rover has looked; unseen ground is assumed flat.
    slope = np.where(belief.seen, world.slope, 0.0)
    cost = 1.0 + cfg.w_slope * np.clip(slope / MAX_SLOPE_DEG, 0, 1)
    blocked = belief.seen & (world.slope > MAX_SLOPE_DEG)
    if use_semantics:
        risk = belief.probs @ np.asarray(cfg.risk, np.float32)
        unc = np.where(belief.known, belief.unc, cfg.unknown_unc)
        cost += np.where(belief.known, cfg.w_terrain * risk, 0.0) + cfg.w_unc * unc
        likely_hazard = (belief.probs[..., SAND] >= cfg.p_block) | (belief.probs[..., ROCK] >= cfg.p_block)
        blocked |= belief.known & likely_hazard
    return cost.astype(np.float32), blocked


def explain(world: World, belief: Belief, cell: tuple[int, int], cfg: CostConfig = CostConfig()) -> str:
    """One-line reason for a cell's cost, for the UI."""
    from .data import CLASSES
    if not belief.seen[cell]:
        return "unseen"
    parts = [f"slope {world.slope[cell]:.0f} deg"]
    if belief.known[cell]:
        p = belief.probs[cell]
        k = int(p.argmax())
        parts.append(f"{CLASSES[k]} P={p[k]:.2f}, uncertainty {belief.unc[cell]:.2f}")
    return ", ".join(parts)
