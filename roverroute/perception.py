"""What the rover knows: cells it has seen within its sensor radius.

Modes
- geometry: senses slope only, no idea what the ground is made of (baseline)
- model:    slope + the terrain model's class probabilities (RedRoute)
- oracle:   slope + the true class (upper bound; the only mode that reads truth)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SOIL, UNLABELLED
from .world import SIZE, World

MODES = ("geometry", "model", "oracle")
SENSOR_RADIUS = 6   # cells = 3 m


@dataclass
class Belief:
    seen: np.ndarray     # (60, 60) bool, geometry observed
    known: np.ndarray    # (60, 60) bool, terrain class observed
    probs: np.ndarray    # (60, 60, 4) class probabilities where known
    unc: np.ndarray      # (60, 60) normalised entropy in [0, 1] where known

    @classmethod
    def empty(cls) -> "Belief":
        return cls(np.zeros((SIZE, SIZE), bool), np.zeros((SIZE, SIZE), bool),
                   np.full((SIZE, SIZE, 4), 0.25, np.float32), np.ones((SIZE, SIZE), np.float32))


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-9, 1.0)
    return (-(p * np.log(p)).sum(axis=-1) / np.log(p.shape[-1])).astype(np.float32)


def visible(cell: tuple[int, int], radius: int = SENSOR_RADIUS) -> np.ndarray:
    rows, cols = np.ogrid[0:SIZE, 0:SIZE]
    return (rows - cell[0]) ** 2 + (cols - cell[1]) ** 2 <= radius ** 2


def observe(world: World, belief: Belief, cell: tuple[int, int], mode: str) -> np.ndarray:
    """Update `belief` with everything visible from `cell`; returns the visible mask."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    view = visible(cell)
    belief.seen |= view
    if mode == "model":
        belief.probs[view] = world.probs[view]
    elif mode == "oracle":
        truth = np.where(world.truth == UNLABELLED, SOIL, world.truth)[view]
        belief.probs[view] = np.eye(4, dtype=np.float32)[truth]
    if mode != "geometry":
        belief.known |= view
        belief.unc[view] = entropy(belief.probs[view])
    return view
