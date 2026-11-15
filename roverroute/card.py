"""Route decision card: why the rover took a longer route than the plain shortest one.

Uses only the rover's belief, never the true terrain.
"""
from __future__ import annotations

from .costmap import CostConfig, cost_grid
from .data import CLASSES, ROCK, SAND
from .perception import Belief
from .planner import path_length, plan
from .world import CELL_M, World


def decision_card(world: World, belief: Belief, start, goal, chosen: list,
                  cfg: CostConfig = CostConfig()) -> dict:
    cost, blocked = cost_grid(world, belief, use_semantics=False, cfg=cfg)
    blocked[start] = False
    shortest = plan(cost, blocked, start, goal)
    chosen_m = path_length(chosen, CELL_M)
    card = {"chosen_m": round(chosen_m, 1), "shortest_m": None, "extra_m": None,
            "extra_pct": None, "avoided_class": None, "avoided_cells": 0, "avoided_mean_p": None}
    if shortest is None:
        card["summary"] = "No terrain-blind route exists to compare against."
        return card
    shortest_m = path_length(shortest, CELL_M)
    extra = max(chosen_m - shortest_m, 0.0)
    card.update(shortest_m=round(shortest_m, 1), extra_m=round(extra, 1),
                extra_pct=round(100 * extra / shortest_m, 1) if shortest_m else 0.0)

    on_chosen = set(chosen)
    avoided = [c for c in shortest if c not in on_chosen and belief.known[c]]
    hazard_cells = {k: [c for c in avoided if belief.probs[c].argmax() == k] for k in (SAND, ROCK)}
    k = max(hazard_cells, key=lambda key: len(hazard_cells[key]))
    cells = hazard_cells[k]
    if extra < 0.05 or not cells:
        card["summary"] = ("Took the shortest route; nothing needed avoiding." if extra < 0.05
                           else f"Took a {extra:.1f} m longer route to stay clear of uncertain ground.")
        return card
    mean_p = float(sum(belief.probs[c][k] for c in cells) / len(cells))
    card.update(avoided_class=CLASSES[k], avoided_cells=len(cells), avoided_mean_p=round(mean_p, 2))
    card["summary"] = (f"Took a {extra:.1f} m ({card['extra_pct']:.0f}%) longer route to avoid "
                       f"{len(cells)} cell{'' if len(cells) == 1 else 's'} of predicted {CLASSES[k]} (mean P={mean_p:.2f}).")
    return card
