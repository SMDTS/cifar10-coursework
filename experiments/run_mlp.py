"""<Exercise N>: <short description>."""
from torch import nn

from src import config
from src.data import get_loaders
from src.models.mlp import MLP                     # CHANGE: import your model
from src.train import evaluate, fit, set_seed
from src.utils import count_parameters, plot_loss_curves, save_results

EXPERIMENT_NAME = "mlp"                            # CHANGE
PLOT_TITLE = "MLP: training vs validation loss"    # CHANGE
AUGMENT = False                                    # CHANGE (True only for augmentation)


def main():
    set_seed(config.SEED)
    train_loader, train_eval_loader, val_loader, test_loader = get_loaders(augment=AUGMENT)

    model = MLP()                                  # CHANGE: your model class
    num_params = count_parameters(model)
    model, history = fit(model, train_loader, val_loader)

    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = evaluate(model, train_eval_loader, criterion, config.DEVICE)
    val_loss, val_acc = evaluate(model, val_loader, criterion, config.DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, criterion, config.DEVICE)

    metrics = {
        "params": num_params,
        "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
        "gap": train_acc - test_acc,
        "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
        "best_epoch": history["best_epoch"],
    }
    save_results(EXPERIMENT_NAME, metrics, history)
    plot_loss_curves(history, PLOT_TITLE, f"{EXPERIMENT_NAME}_loss.png")

    print(f"\nParameters: {num_params:,}")
    print(f"Train acc {train_acc:.4f} | Val acc {val_acc:.4f} | "
          f"Test acc {test_acc:.4f} | Gap {train_acc - test_acc:.4f}")


if __name__ == "__main__":
    main()