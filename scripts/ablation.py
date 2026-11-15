"""Do the model's ML choices change how the rover drives? Same 100 worlds, model rover only.

    python -m scripts.ablation --n 100

Varies: calibrated (T=1.52) vs raw (T=1) probabilities, the sand/rock block threshold,
and the uncertainty cost term. Writes results/ablation.{json,md}.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from roverroute.costmap import CostConfig
from roverroute.data import DATA_DIR, TILE
from roverroute.metrics import bootstrap_ci, score
from roverroute.rover import run_mission
from roverroute.world import build_world

OUT = Path(__file__).resolve().parent.parent / "results"
BASE = CostConfig()
CONFIGS = [   # (name, prediction file, cost config)
    ("calibrated, block P>=0.5 (shipped)", "preds20.npz", BASE),
    ("raw T=1, block P>=0.5", "preds20_T1.npz", BASE),
    ("calibrated, block P>=0.3", "preds20.npz", replace(BASE, p_block=0.3)),
    ("raw T=1, block P>=0.3", "preds20_T1.npz", replace(BASE, p_block=0.3)),
    ("calibrated, block P>=0.7", "preds20.npz", replace(BASE, p_block=0.7)),
    ("raw T=1, block P>=0.7", "preds20_T1.npz", replace(BASE, p_block=0.7)),
    ("calibrated, no uncertainty cost", "preds20.npz", replace(BASE, w_unc=0.0)),
]


def load_probs(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA_DIR / name) as z:
        return {str(i): p.astype(np.float32) for i, p in zip(z["ids"], z["probs"])}


def with_probs(world, probs: dict[str, np.ndarray]):
    """Same world, but every tile's prediction swapped for the given file's."""
    for tr, row in enumerate(world.tile_ids):
        for tc, tile in enumerate(row):
            world.probs[tr * TILE:(tr + 1) * TILE, tc * TILE:(tc + 1) * TILE] = probs[tile]
    return world


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    files = {f: load_probs(f) for f in {c[1] for c in CONFIGS}}
    rows = []
    for name, file, cfg in CONFIGS:
        res = []
        for seed in range(args.n):
            world = with_probs(build_world(seed), files[file])
            mission = run_mission(world, "model", cfg)
            res.append(score(world, mission.trajectory, mission.reached))
        success = [r["success"] for r in res]
        entries = [r["hazard_entries"] for r in res]
        ratios = [r["route_ratio"] for r in res if r["route_ratio"]]
        rows.append({"config": name, "success": float(np.mean(success)), "success_ci95": bootstrap_ci(success),
                     "reached": float(np.mean([r["reached"] for r in res])),
                     "hazard_entries": float(np.mean(entries)),
                     "route_ratio": float(np.mean(ratios)) if ratios else None})
        print(rows[-1], flush=True)

    lines = ["# Ablation: model choices vs rover outcome", "",
             f"Model rover only, seeds 0..{args.n - 1}, same worlds for every row "
             "(`python -m scripts.ablation`).", "",
             "| Config | Success [95% CI] | Reached goal | Hazard entries / mission | Route ratio |",
             "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lo, hi = r["success_ci95"]
        ratio = f"{r['route_ratio']:.3f}" if r["route_ratio"] else "n/a"
        lines.append(f"| {r['config']} | {r['success']:.2f} [{lo:.2f}, {hi:.2f}] | {r['reached']:.2f} | "
                     f"{r['hazard_entries']:.2f} | {ratio} |")
    (OUT / "ablation.json").write_text(json.dumps({"n": args.n, "rows": rows}, indent=1))
    (OUT / "ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
