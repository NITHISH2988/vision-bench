"""Training and evaluation loops shared by the CLI and tests."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _run_batches(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, targets, predictions = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach()) * len(labels))
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    if not targets:
        raise ValueError("DataLoader produced no examples")
    return {
        "loss": sum(losses) / len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "targets": targets,
        "predictions": predictions,
    }


def evaluate(model, loader: DataLoader, device, class_names=None):
    result = _run_batches(model, loader, nn.CrossEntropyLoss(), device)
    count = len(class_names) if class_names else int(max(result["targets"]) + 1)
    labels = list(range(count))
    result["confusion_matrix"] = confusion_matrix(
        result["targets"], result["predictions"], labels=labels
    ).tolist()
    result["per_class"] = classification_report(
        result["targets"],
        result["predictions"],
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return result


def train_model(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    model_name: str,
    class_names: list[str],
    epochs: int,
    learning_rate: float,
    device,
    checkpoint_path: Path,
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    model = build_model(model_name, len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = _run_batches(model, train_loader, loss_fn, device, optimizer)
        validation_metrics = evaluate(model, validation_loader, device, class_names)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(
                {
                    "model_name": model_name,
                    "class_names": class_names,
                    "epoch": epoch,
                    "validation_accuracy": best_accuracy,
                    "model_state_dict": model.state_dict(),
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        print(json.dumps(history[-1]))
    return history


def save_training_artifacts(
    history: list[dict], history_path: Path, plot_path: Path
) -> None:
    """Persist epoch metrics and render loss/accuracy curves without a GUI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(
        epochs, [row["validation_loss"] for row in history], label="validation"
    )
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(
        epochs, [row["validation_accuracy"] for row in history], label="validation"
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
