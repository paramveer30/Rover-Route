"""Find seeds where the terrain-blind rover hits a hazard and the model rover visibly detours.

    python -m scripts.pick_demo --n 100
"""
from __future__ import annotations

import argparse

from roverroute.card import decision_card
from roverroute.metrics import score
from roverroute.rover import run_mission
from roverroute.world import build_world


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    found = []
    for seed in range(args.n):
        world = build_world(seed)
        blind, smart = run_mission(world, "geometry"), run_mission(world, "model")
        b, s = score(world, blind.trajectory, blind.reached), score(world, smart.trajectory, smart.reached)
        if b["hazard_entries"] == 0 or not s["success"]:
            continue
        card = decision_card(world, smart.belief, world.start, world.goal, smart.trajectory)
        if card["avoided_cells"]:
            found.append((b["hazard_entries"], card["avoided_cells"], seed, card["summary"]))
    for entries, cells, seed, summary in sorted(found, reverse=True)[:10]:
        print(f"seed {seed:3d}: geometry hit {entries} hazard(s); model: {summary}")


if __name__ == "__main__":
    main()
