"""Run the model on every test image, project its probabilities top-down, and write the rover's tiles.

    python -m ml.precompute                                   # data/preds20.npz + val/test logits
    python -m ml.precompute --temperature 1 --out data/preds20_T1.npz --no-val   # raw model, for the ablation

Each NAVCAM frame is warped onto a flat 20 x 20 cell patch with the homography stored in
data/tiles.npz (a flat-ground approximation, no camera calibration). Probabilities are
softmax(logits / T), with T from results/ml/calibration.json unless given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import (LOGITS_DIR, PREDS_NPZ, RESULTS, TILES_NPZ, USABLE_TILES, clean_label, load_test,
                   load_train, tile_id_for, train_val_split)
from .model import load_weights, preprocess

CELLS, OUT_PX, SOURCE_PX = 20, 160, 1024
LOGIT_SIZE = 64   # logits kept for calibration are downsampled to 64 x 64 to stay small


def warp_probs(probs_hw4: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp (h, w, 4) probabilities to the 160 px top-down patch, average 8 x 8 blocks into
    20 x 20 cells, renormalise. Cells with no image behind them become uniform."""
    h, w = probs_hw4.shape[:2]
    matrix = np.asarray(H, np.float64) @ np.diag([SOURCE_PX / w, SOURCE_PX / h, 1.0])
    patch = cv2.warpPerspective(probs_hw4.astype(np.float32), matrix, (OUT_PX, OUT_PX),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    block = OUT_PX // CELLS
    cells = patch.reshape(CELLS, block, CELLS, block, 4).mean(axis=(1, 3))
    mass = cells.sum(-1, keepdims=True)
    return np.divide(cells, mass, out=np.full_like(cells, 0.25), where=mass > 0)


@torch.inference_mode()
def infer(model, rows, device, batch_size, matrices=None, T=1.0):
    """Raw 64 px logits + labels for calibration, and optionally the warped 20 x 20 tile probabilities."""
    n = len(rows)
    logits64 = np.empty((n, 4, LOGIT_SIZE, LOGIT_SIZE), np.float16)
    labels64 = np.empty((n, LOGIT_SIZE, LOGIT_SIZE), np.uint8)
    tiles = np.empty((n, CELLS, CELLS, 4), np.float16) if matrices is not None else None
    for start in range(0, n, batch_size):
        batch = [rows[i] for i in range(start, min(start + batch_size, n))]
        logits = model(torch.stack([preprocess(r["image"]) for r in batch]).to(device)).float()
        end = start + len(batch)
        logits64[start:end] = F.interpolate(logits, size=(LOGIT_SIZE,) * 2, mode="bilinear",
                                            align_corners=False).cpu().numpy()
        labels64[start:end] = [cv2.resize(clean_label(r["label_mask"], r["rover_mask"], r["range_mask"]),
                                          (LOGIT_SIZE,) * 2, interpolation=cv2.INTER_NEAREST) for r in batch]
        if matrices is not None:
            probs = torch.softmax(logits / T, 1).permute(0, 2, 3, 1).cpu().numpy()
            for k, p in enumerate(probs):
                tiles[start + k] = warp_probs(p, matrices[start + k])
        print(f"inferred {end}/{n}", flush=True)
    return logits64, labels64, tiles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--out", type=Path, default=PREDS_NPZ)
    p.add_argument("--no-val", action="store_true", help="skip validation logits (only needed by ml.calibrate)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    calib = RESULTS / "calibration.json"
    T = args.temperature if args.temperature is not None else (
        json.loads(calib.read_text())["T"] if calib.exists() else 1.0)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    model = load_weights(device=device)

    rows = load_test()
    with np.load(TILES_NPZ) as z:
        ids, matrices, truth = z["ids"], z["H"], z["label20"]
    assert list(ids) == [tile_id_for(i) for i in range(len(rows))], "tiles.npz must list every test image in order"
    print(f"device={device} T={T:.4f}", flush=True)
    logits, labels, probs = infer(model, rows, device, args.batch_size, matrices, T)

    usable = np.isin(ids, json.loads(USABLE_TILES.read_text()))
    valid = (truth != 255) & usable[:, None, None]
    agreement = float((probs.argmax(-1)[valid] == truth[valid]).mean())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=ids, probs=probs, temperature=np.float64(T),
                        tile_agreement=np.float64(agreement), labelled_cells=np.int64(valid.sum()))
    print(f"wrote {args.out}; cell agreement with labels on usable tiles {agreement:.1%}")

    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LOGITS_DIR / "test_logits.npz", logits=logits, labels=labels)
    if not args.no_val:
        _, val_rows = train_val_split(load_train(6500), 500)
        v_logits, v_labels, _ = infer(model, val_rows, device, args.batch_size)
        np.savez_compressed(LOGITS_DIR / "val_logits.npz", logits=v_logits, labels=v_labels)
        print(f"wrote logits to {LOGITS_DIR}")


if __name__ == "__main__":
    main()
