from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure the parent directory exists, then pretty-dump the obj as JSON to path
def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# Draws train/val curves across epochs, then overlays two reference horizontal lines: 
#   the training data entropy and validation data entropy (the “shot-noise floor” you want to hit). 
# Saves PNG and frees the figure
def _plot_losses(train, val, data_entropy, v_data_entropy, title, out_png: Path):
    fig = plt.figure(figsize=(8,5))
    xs = list(range(len(train)))
    plt.plot(xs, train, label="Training Likelihood")
    plt.plot(xs, val, label="Validation Likelihood")
    plt.axhline(data_entropy, color="black", linestyle="--", linewidth=1.0)
    plt.axhline(v_data_entropy, color="blue", linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.ylabel("Cross Entropy")
    plt.xlabel("Epoch")
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# Generic boxplot helper (used for fidelities)
def _plot_box(vals, title, out_png: Path):
    fig = plt.figure(figsize=(6,5))
    plt.boxplot(np.array(vals))
    plt.ylabel(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# Orchestrates saving and plotting
# Expected keys are commented
# If available, it plots overall fidelities and (optionally) U-specific fidelities
def write_report(run_dir: Path, metrics: dict, prefix="run"):
    # metrics keys: epoch_losses, epoch_val_losses, data_entropy, v_data_entropy,
    # fidelities, fidelities_U (optional)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, run_dir / f"{prefix}_metrics.json")

    _plot_losses(metrics["epoch_losses"], metrics["epoch_val_losses"],
                 metrics["data_entropy"], metrics["v_data_entropy"],
                 metrics.get("title", "ptnt"), run_dir / f"{prefix}_losses.png")
    if "fidelities" in metrics:
        _plot_box(metrics["fidelities"], "Reconstruction Fidelity", run_dir / f"{prefix}_fidelities.png")
    if "fidelities_U" in metrics:
        _plot_box(metrics["fidelities_U"], "Reconstruction Fidelity (U)", run_dir / f"{prefix}_fidelities_U.png")
