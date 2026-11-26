"""Regenerate loss/accuracy curves for every experiment from results/*.json."""
import json
import os

from src import config
from src.utils import plot_training_curves

TITLES = {
    "mlp": "MLP",
    "cnn_baseline": "Baseline CNN",
    "cnn_dropout": "CNN + dropout",
    "cnn_augment": "CNN + augmentation",
}


def main():
    for name, title in TITLES.items():
        path = os.path.join(config.RESULTS_DIR, f"{name}.json")
        if not os.path.exists(path):
            print(f"Skipping {name}: no results yet")
            continue
        with open(path) as f:
            history = json.load(f)["history"]
        plot_training_curves(history, title, f"{name}_curves.png")
        print(f"Saved results/{name}_curves.png")


if __name__ == "__main__":
    main()