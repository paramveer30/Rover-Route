"""Score the frozen model on the 322-image expert test split (and the validation split).

    python -m ml.evaluate

Logits are upsampled to the native 1024 px mask before argmax. IoU is computed from one
pixel-level confusion matrix over the whole split, not averaged per image. Writes
results/ml/seg_metrics.json and results/ml/confusion.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from .data import CLASS_NAMES, RESULTS, WEIGHTS, clean_label, load_test, load_train, train_val_split
from .model import INPUT_SIZE, load_weights, preprocess


def confusion(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """4x4 counts, truth rows x prediction columns, ignoring unlabelled pixels."""
    ok = labels != 255
    return np.bincount(labels[ok].astype(np.int64) * 4 + pred[ok], minlength=16).reshape(4, 4)


def metrics(cm: np.ndarray) -> dict:
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.divide(inter, union, out=np.full(4, np.nan), where=union > 0)
    return {"miou": float(np.nanmean(iou)),
            "iou": dict(zip(CLASS_NAMES, iou.tolist())),
            "pixel_acc": float(inter.sum() / cm.sum()),
            "class_share": dict(zip(CLASS_NAMES, (cm.sum(1) / cm.sum()).tolist())),
            "labelled_pixels": int(cm.sum()),
            "confusion_matrix": cm.tolist()}


@torch.inference_mode()
def evaluate_split(model, rows, device, batch_size=16):
    cm = np.zeros((4, 4), np.int64)
    for start in range(0, len(rows), batch_size):
        batch = [rows[i] for i in range(start, min(start + batch_size, len(rows)))]
        logits = model(torch.stack([preprocess(r["image"]) for r in batch]).to(device))
        for j, r in enumerate(batch):
            label = clean_label(r["label_mask"], r["rover_mask"], r["range_mask"])
            up = F.interpolate(logits[j:j + 1], size=label.shape, mode="bilinear", align_corners=False)
            cm += confusion(label, up.argmax(1)[0].cpu().numpy())
        print(f"evaluated {start + len(batch)}/{len(rows)}", flush=True)
    return metrics(cm)


def image_hashes(rows) -> list[str]:
    from datasets import Image
    encoded = rows.select_columns(["image"]).cast_column("image", Image(decode=False))
    return [hashlib.sha256(r["image"]["bytes"] or open(r["image"]["path"], "rb").read()).hexdigest()
            for r in encoded]


@torch.inference_mode()
def cpu_latency_ms(model, rows, n=50) -> dict:
    """Batch-1 forward pass on CPU (preprocessing excluded): can a rover run this?"""
    model = model.cpu()
    inputs = [preprocess(rows[i]["image"]).unsqueeze(0) for i in range(n)]
    for x in inputs[:5]:
        model(x)
    times = []
    for x in inputs:
        t = perf_counter()
        model(x)
        times.append((perf_counter() - t) * 1000)
    p50, p95 = np.percentile(times, [50, 95])
    return {"p50": float(p50), "p95": float(p95), "threads": torch.get_num_threads()}


def save_confusion_plot(cm, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = [n.replace("_", " ") for n in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.set(xlabel="predicted", ylabel="expert label", title="AI4Mars test_min3 (322 images), row-normalised")
    for r in range(4):
        for c in range(4):
            ax.text(c, r, f"{norm[r, c]:.1%}", ha="center", va="center", color="white" if norm[r, c] > 0.5 else "black")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    model = load_weights(device=device)
    test_rows = load_test()
    subset = load_train(6500)
    train_rows, val_rows = train_val_split(subset, 500)

    print("test split", flush=True)
    test = evaluate_split(model, test_rows, device)
    print("validation split", flush=True)
    val = evaluate_split(model, val_rows, device)

    test_hashes = set(image_hashes(test_rows))
    overlap = len(set(image_hashes(subset)) & test_hashes)
    result = {"split": "test_min3", "n_images": len(test_rows), "input_size": INPUT_SIZE, **test,
              "val_miou": val["miou"], "val_iou": val["iou"], "val_pixel_acc": val["pixel_acc"],
              "val_n_images": len(val_rows),
              "train_test_overlap_images": overlap,
              "cpu_latency_ms": cpu_latency_ms(model, test_rows),
              "weights_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "seg_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    save_confusion_plot(test["confusion_matrix"], RESULTS / "confusion.png")
    print(json.dumps({k: result[k] for k in ("miou", "iou", "pixel_acc", "val_miou", "train_test_overlap_images",
                                             "cpu_latency_ms")}, indent=2))


if __name__ == "__main__":
    main()
