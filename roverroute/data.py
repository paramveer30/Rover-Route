"""Load the real AI4Mars tile data: ground-truth labels and model predictions.

Each tile is a top-down 20 x 20 cell patch made from one Curiosity NAVCAM image.
Classes: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 unlabelled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLASSES = ("soil", "bedrock", "sand", "big rock")
SOIL, BEDROCK, SAND, ROCK, UNLABELLED = 0, 1, 2, 3, 255
TILE = 20


@dataclass(frozen=True)
class TileData:
    ids: tuple[str, ...]          # usable tile ids, e.g. "m3_0046"
    labels: np.ndarray            # (n, 20, 20) uint8 ground truth
    probs: np.ndarray             # (n, 20, 20, 4) float32 model probabilities
    temperature: float            # calibration temperature used for probs


@lru_cache(maxsize=1)
def load_tiles() -> TileData:
    usable = json.loads((DATA_DIR / "usable_tiles.json").read_text())
    with np.load(DATA_DIR / "tiles.npz") as t, np.load(DATA_DIR / "preds20.npz") as p:
        ids = [str(i) for i in t["ids"]]
        if ids != [str(i) for i in p["ids"]]:
            raise ValueError("tiles.npz and preds20.npz list different tiles")
        index = [ids.index(i) for i in usable]
        return TileData(
            ids=tuple(usable),
            labels=t["label20"][index].astype(np.uint8),
            probs=p["probs"][index].astype(np.float32),
            temperature=float(p["temperature"]),
        )


def frame_path(tile_id: str) -> Path:
    return DATA_DIR / "frames" / f"{tile_id}.jpg"


def overlay_path(tile_id: str) -> Path:
    return DATA_DIR / "overlays" / f"{tile_id}.png"
