"""Helpers for parameter counts, loss plots and saving/loading results."""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from src import config


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss_curves(history, title, filename):
    """Plot training vs validation loss per epoch and save to results/."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    if "best_epoch" in history:
        plt.axvline(history["best_epoch"], linestyle="--", color="grey",
                    label="Best validation epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, filename), dpi=150)
    plt.close()


def save_results(name, metrics, history):
    """Save final metrics and per-epoch history to results/<name>.json."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump({"name": name, **metrics, "history": history}, f, indent=2)


def load_all_results():
    """Load every results/*.json into a DataFrame (without the history) for tables."""
    rows = []
    for filename in sorted(os.listdir(config.RESULTS_DIR)):
        if filename.endswith(".json"):
            with open(os.path.join(config.RESULTS_DIR, filename)) as f:
                data = json.load(f)
            data.pop("history", None)
            rows.append(data)
    return pd.DataFrame(rows)

def plot_training_curves(history, title, filename):
    """Side-by-side training vs validation loss and accuracy per epoch."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history["train_loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Validation")
    ax_loss.set(title=f"{title}: Loss", xlabel="Epoch", ylabel="Cross-entropy loss")

    ax_acc.plot(epochs, history["train_acc"], label="Train")
    ax_acc.plot(epochs, history["val_acc"], label="Validation")
    ax_acc.set(title=f"{title}: Accuracy", xlabel="Epoch", ylabel="Accuracy")

    for ax in (ax_loss, ax_acc):
        if "best_epoch" in history:
            ax.axvline(history["best_epoch"], linestyle="--", color="grey",
                       label="Best validation epoch")
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, filename), dpi=150)
    plt.close(fig)