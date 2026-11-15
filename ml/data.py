"""AI4Mars loading, label cleaning, the fixed train/val split, and file locations.

Dataset: hassanjbara/AI4MARS on Hugging Face (Curiosity NAVCAM, 1024 x 1024).
Labels: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 no label.
`test_min3` (322 images, labels agreed by at least 3 experts) is the test set. It is never
used for training, checkpoint selection or calibration.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "ml"                      # small, committed: metrics, logs, plots
TILES_NPZ = REPO / "data" / "tiles.npz"                # tile ids, homographies, 20x20 labels
PREDS_NPZ = REPO / "data" / "preds20.npz"              # model output the rover uses
USABLE_TILES = REPO / "data" / "usable_tiles.json"
WEIGHTS = Path(os.environ.get("RR_WEIGHTS", REPO / "ml" / "weights" / "unet_r34_fp16.pt"))
ARTIFACTS = Path(os.environ.get("RR_ARTIFACTS", REPO / "artifacts"))   # big, gitignored
DATA_DIR = Path(os.environ.get("RR_DATA", ARTIFACTS / "data"))       # cached train subset
CKPT_DIR = ARTIFACTS / "checkpoints"
LOGITS_DIR = ARTIFACTS / "logits"

HF_NAME = "hassanjbara/AI4MARS"
TEST_SPLIT = "test_min3"
CLASS_NAMES = ("soil", "bedrock", "sand", "big_rock")
IGNORE = 255


def tile_id_for(index: int) -> str:
    return f"m3_{index:04d}"


def clean_label(label, rover_mask, range_mask) -> np.ndarray:
    """Mark rover-body pixels and pixels beyond 30 m as IGNORE.

    Expert test labels already do this; crowdsourced train labels may not.
    """
    out = np.asarray(label, np.uint8).copy()
    if rover_mask is not None:
        out[np.asarray(rover_mask) > 0] = IGNORE
    if range_mask is not None:
        out[np.asarray(range_mask) > 0] = IGNORE
    out[out > 3] = IGNORE
    return out


def to_rgb(image) -> np.ndarray:
    img = np.asarray(image, np.uint8)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"unsupported image shape {img.shape}")
    return img


def load_test():
    from datasets import load_dataset
    return load_dataset(HF_NAME, split=TEST_SPLIT)


def load_train(n: int = 6500, seed: int = 0):
    """n labelled train rows, downloaded once into DATA_DIR/train_<n>."""
    from datasets import load_dataset, load_from_disk
    path = DATA_DIR / f"train_{n}"
    if path.exists():
        return load_from_disk(str(path))
    ds = load_dataset(HF_NAME, split=f"train[:{int(n * 1.3)}]")
    ds = ds.filter(lambda r: bool(r["has_labels"]) and r["label_mask"] is not None)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    return ds


def train_val_split(ds, n_val: int = 500, seed: int = 0):
    """Fixed split. Validation picks the checkpoint and fits the temperature."""
    split = ds.train_test_split(test_size=n_val, seed=seed)
    return split["train"], split["test"]
