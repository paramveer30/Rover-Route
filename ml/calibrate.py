"""Temperature scaling: fit one scalar T on validation logits so the probabilities are honest.

    python -m ml.calibrate         # needs logits from `python -m ml.precompute`

softmax(logits / T). T > 1 softens an overconfident model. T is fitted by minimising
validation NLL; the test numbers are only reported. Writes results/ml/calibration.json
and results/ml/reliability.png.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .data import LOGITS_DIR, RESULTS

N_BINS = 15


def flatten(logits: np.ndarray, labels: np.ndarray):
    """(N,4,H,W), (N,H,W) -> (M,4) logits and (M,) labels over labelled pixels."""
    lg = np.moveaxis(logits, 1, -1).reshape(-1, logits.shape[1]).astype(np.float32)
    lb = labels.reshape(-1)
    keep = lb != 255
    return lg[keep], lb[keep].astype(np.int64)


def nll(logits, labels, T=1.0) -> float:
    return float(torch.nn.functional.cross_entropy(torch.from_numpy(logits.astype(np.float64)) / T,
                                                   torch.from_numpy(labels)))


def fit_temperature(logits, labels) -> float:
    lg, lb = torch.from_numpy(logits.astype(np.float64)), torch.from_numpy(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)   # optimise log T so T stays > 0
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def reliability(logits, labels, T=1.0):
    """Per confidence bin: mean confidence, accuracy, pixel count."""
    p = torch.softmax(torch.from_numpy(logits.astype(np.float64)) / T, 1).numpy()
    conf, correct = p.max(1), p.argmax(1) == labels
    idx = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    count = np.bincount(idx, minlength=N_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.bincount(idx, conf, N_BINS) / count,
                np.bincount(idx, correct.astype(float), N_BINS) / count, count)


def ece(logits, labels, T=1.0) -> float:
    """Expected calibration error: confidence-vs-accuracy gap, weighted by bin size."""
    mc, acc, count = reliability(logits, labels, T)
    m = count > 0
    return float((count[m] / count.sum() * np.abs(acc[m] - mc[m])).sum())


def plot(val, test, T, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    for ax, (name, (lg, lb)) in zip(axes, (("validation", val), ("test", test))):
        for t, label, colour in ((1.0, "raw (T=1)", "tab:red"), (T, f"calibrated (T={T:.2f})", "tab:blue")):
            mc, acc, count = reliability(lg, lb, t)
            m = count > 0
            ax.plot(mc[m], acc[m], "o-", color=colour, label=f"{label}, ECE={ece(lg, lb, t):.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=name, xlabel="confidence")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    splits = {}
    for name in ("val", "test"):
        with np.load(LOGITS_DIR / f"{name}_logits.npz") as z:
            splits[name] = flatten(z["logits"], z["labels"])
    T = fit_temperature(*splits["val"])
    out = {"T": T}
    for name, (lg, lb) in splits.items():
        out.update({f"ece_{name}_before": ece(lg, lb), f"ece_{name}_after": ece(lg, lb, T),
                    f"nll_{name}_before": nll(lg, lb), f"nll_{name}_after": nll(lg, lb, T),
                    f"n_pixels_{name}": int(len(lb))})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=2) + "\n")
    plot(splits["val"], splits["test"], T, RESULTS / "reliability.png")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
