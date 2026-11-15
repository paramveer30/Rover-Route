"""Train the terrain U-Net on 6,000 AI4Mars images; the 500-image validation split picks the checkpoint.

    python -m ml.train                 # full run, ~25 min on one GPU
    python -m ml.train --plot-only     # redraw results/ml/learning_curve.png from the log

The test split is never loaded here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from .data import ARTIFACTS, CKPT_DIR, CLASS_NAMES, IGNORE, RESULTS, WEIGHTS, clean_label, load_train, to_rgb, train_val_split
from .model import INPUT_SIZE, build_model, load_weights, preprocess

LOG = RESULTS / "train_log.csv"


class PreparedDataset(Dataset):
    """Decode and resize every row once into memory-mapped arrays, then train from those."""

    def __init__(self, rows, directory: Path, size: int, augment: bool):
        self.size, self.augment = size, augment
        directory.mkdir(parents=True, exist_ok=True)
        img_path, lbl_path = directory / "images.npy", directory / "labels.npy"
        if not (img_path.exists() and lbl_path.exists() and len(np.load(lbl_path, mmap_mode="r")) == len(rows)):
            images = np.lib.format.open_memmap(img_path, "w+", np.uint8, (len(rows), size, size, 3))
            labels = np.lib.format.open_memmap(lbl_path, "w+", np.uint8, (len(rows), size, size))
            for i, row in enumerate(rows):
                label = clean_label(row["label_mask"], row["rover_mask"], row["range_mask"])
                images[i] = cv2.resize(to_rgb(row["image"]), (size, size), interpolation=cv2.INTER_AREA)
                labels[i] = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
                if (i + 1) % 500 == 0:
                    print(f"prepared {directory.name}: {i + 1}/{len(rows)}", flush=True)
            images.flush(), labels.flush()
            del images, labels
        self.images = np.load(img_path, mmap_mode="r")
        self.labels = np.load(lbl_path, mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image, label = np.array(self.images[i]), np.array(self.labels[i])
        if self.augment:   # horizontal flip + small brightness/contrast jitter
            if random.random() < 0.5:
                image, label = image[:, ::-1].copy(), label[:, ::-1].copy()
            contrast, brightness = random.uniform(0.9, 1.1), random.uniform(-0.05, 0.05) * 255
            image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness, 0, 255).astype(np.uint8)
        return preprocess(image, self.size), torch.from_numpy(label.astype(np.int64))


def class_weights(dataset: PreparedDataset):
    """Inverse-sqrt frequency weights, clipped to [1, 20]; big rock is under 1% of pixels."""
    counts = np.zeros(4, np.int64)
    for label in dataset.labels:
        counts += np.bincount(label[label < 4].ravel(), minlength=4)
    freq = counts / counts.sum()
    return torch.tensor(np.clip(1 / np.sqrt(np.maximum(freq, 1e-12)), 1, 20), dtype=torch.float32), counts


def iou_from_confusion(cm: torch.Tensor):
    cm = cm.cpu().numpy().astype(np.float64)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.divide(inter, union, out=np.full(4, np.nan), where=union > 0)
    return float(np.nanmean(iou)), iou


def plot_log(log: Path = LOG, out: Path = RESULTS / "learning_curve.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with log.open() as f:
        rows = list(csv.DictReader(f))
    epoch = [int(r["epoch"]) for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(epoch, [float(r["train_loss"]) for r in rows], "o-")
    a.set(title="training loss (CE + Dice)", xlabel="epoch")
    for name in CLASS_NAMES:
        b.plot(epoch, [float(r[f"iou_{name}"]) for r in rows], label=name.replace("_", " "))
    b.plot(epoch, [float(r["val_miou"]) for r in rows], "k-", lw=2.5, label="mIoU")
    b.set(title="validation IoU", xlabel="epoch", ylim=(0, 1))
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    if args.plot_only:
        return plot_log()

    started = time.perf_counter()
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"device={device}", flush=True)

    train_rows, val_rows = train_val_split(load_train(6500), 500)
    prepared = ARTIFACTS / f"prepared_6500_{INPUT_SIZE}"
    train_ds = PreparedDataset(train_rows, prepared / "train", INPUT_SIZE, augment=True)
    val_ds = PreparedDataset(val_rows, prepared / "val", INPUT_SIZE, augment=False)
    weights, counts = class_weights(train_ds)
    print(f"train={len(train_ds)} val={len(val_ds)} class weights={weights.tolist()}", flush=True)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, pin_memory=amp)
    val_loader = DataLoader(val_ds, args.batch_size, pin_memory=amp)

    model = build_model().to(device)
    ce = torch.nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=IGNORE)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    autocast = (lambda: torch.autocast("cuda")) if amp else nullcontext

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / "best.pt"
    best_iou, best_epoch, best_per_class = -math.inf, 0, None
    fields = ["epoch", "train_loss", "val_miou", *[f"iou_{n}" for n in CLASS_NAMES], "lr", "epoch_seconds", "elapsed_seconds"]
    with LOG.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            total, seen = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                if not (labels != IGNORE).any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(images)
                    loss = ce(logits, labels) + dice(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(images)
                seen += len(images)

            model.eval()
            cm = torch.zeros((4, 4), dtype=torch.int64, device=device)
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    with autocast():
                        pred = model(images).argmax(1)
                    ok = labels != IGNORE
                    cm += torch.bincount(4 * labels[ok] + pred[ok], minlength=16).reshape(4, 4)
            miou, per_class = iou_from_confusion(cm)
            row = dict(epoch=epoch, train_loss=total / seen, val_miou=miou, lr=optimizer.param_groups[0]["lr"],
                       epoch_seconds=time.perf_counter() - t0, elapsed_seconds=time.perf_counter() - started)
            row.update({f"iou_{n}": float(v) for n, v in zip(CLASS_NAMES, per_class)})
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if miou > best_iou:
                best_iou, best_epoch, best_per_class = miou, epoch, per_class
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": miou}, ckpt)
            scheduler.step()
            if epoch >= 15 and epoch - best_epoch >= args.patience:
                print(f"validation plateau; best epoch {best_epoch}", flush=True)
                break

    model.cpu().load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.half().state_dict(), WEIGHTS)
    with torch.inference_mode():   # the exported weights must run on CPU in fp32
        out = load_weights(WEIGHTS)(preprocess(val_ds.images[0]).unsqueeze(0))
    assert out.shape == (1, 4, INPUT_SIZE, INPUT_SIZE) and torch.isfinite(out).all()

    summary = {"best_val_miou": best_iou, "best_epoch": best_epoch,
               "best_class_iou": dict(zip(CLASS_NAMES, best_per_class.tolist())),
               "epochs_completed": epoch, "wall_seconds": time.perf_counter() - started,
               "class_weights": weights.tolist(), "class_counts": counts.tolist(),
               "train_rows": len(train_ds), "val_rows": len(val_ds), "input_size": INPUT_SIZE,
               "seed": args.seed, "weight_bytes": WEIGHTS.stat().st_size}
    (RESULTS / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    plot_log()


if __name__ == "__main__":
    main()
