"""Run every mode on the same seeded worlds and write results/results.{json,md}.

    python -m scripts.run_eval --n 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from roverroute.data import load_tiles
from roverroute.metrics import bootstrap_ci, score
from roverroute.perception import MODES
from roverroute.rover import run_mission
from roverroute.world import build_world

OUT = Path(__file__).resolve().parent.parent / "results"
LABELS = {"geometry": "Geometry only (slope)", "model": "Rover-Route (model)", "oracle": "Oracle (true terrain)"}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def summarise(rows: list[dict]) -> dict:
    success = [r["success"] for r in rows]
    entries = [r["hazard_entries"] for r in rows]
    ratios = [r["route_ratio"] for r in rows if r["route_ratio"] is not None]
    return {
        "n": len(rows),
        "success_rate": float(np.mean(success)),
        "success_ci95": bootstrap_ci(success),
        "reached_rate": float(np.mean([r["reached"] for r in rows])),
        "hazard_entries_mean": float(np.mean(entries)),
        "hazard_entries_ci95": bootstrap_ci(entries),
        "route_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "route_ratio_n": len(ratios),
        "replans_mean": float(np.mean([r["replans"] for r in rows])),
    }


def to_markdown(result: dict) -> str:
    lines = [
        "# Results", "",
        f"{result['n']} seeded worlds (seeds 0..{result['n'] - 1}), every mode on the same worlds. "
        f"git `{result['git']}`, model temperature {result['temperature']:.4f}, {result['seconds']:.1f} s.", "",
        "| Rover | Success [95% CI] | Reached goal | Hazard entries / mission [95% CI] | Route ratio (reached) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        s = result["modes"][mode]
        lo, hi = s["success_ci95"]
        hlo, hhi = s["hazard_entries_ci95"]
        ratio = f"{s['route_ratio_mean']:.3f} (n={s['route_ratio_n']})" if s["route_ratio_mean"] else "n/a"
        lines.append(f"| {LABELS[mode]} | {s['success_rate']:.2f} [{lo:.2f}, {hi:.2f}] | {s['reached_rate']:.2f} | "
                     f"{s['hazard_entries_mean']:.2f} [{hlo:.2f}, {hhi:.2f}] | {ratio} |")
    lines += [
        "", "## Definitions", "",
        "- **Hazard**: true sand, true big rock, or slope over 20 degrees.",
        "- **Hazard entry**: the rover drives onto hazard cells; a continuous run counts once.",
        "- **Success**: reached the goal with 0 hazard entries.",
        "- **Route ratio**: distance driven / shortest route that never touches a true hazard.",
        "- **CI**: 95% percentile bootstrap of the mean.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    started = time.perf_counter()
    per_mode: dict[str, list[dict]] = {m: [] for m in MODES}
    for seed in range(args.n):
        world = build_world(seed)
        for mode in MODES:
            mission = run_mission(world, mode)
            row = score(world, mission.trajectory, mission.reached)
            row.update(seed=seed, replans=mission.replans, stop_reason=mission.stop_reason)
            per_mode[mode].append(row)
    result = {
        "n": args.n, "git": git_sha(), "temperature": load_tiles().temperature,
        "seconds": time.perf_counter() - started,
        "modes": {m: summarise(rows) for m, rows in per_mode.items()},
        "missions": per_mode,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=1))
    (args.out / "results.md").write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == "__main__":
    main()
