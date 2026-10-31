"""Train a CIFAR-10 classifier and evaluate its best validation checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from .data import make_loaders
from .engine import evaluate, save_training_artifacts, set_seed, train_model
from .models import build_model


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch installation cannot access CUDA"
        )
    return device


def train(args) -> dict:
    set_seed(args.seed)
    device = _device(args.device)
    train_loader, validation_loader, test_loader, classes = make_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.workers,
    )
    history = train_model(
        train_loader,
        validation_loader,
        model_name=args.model,
        class_names=classes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    save_training_artifacts(history, args.history, args.plot)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_name"], len(checkpoint["class_names"])).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, test_loader, device, checkpoint["class_names"])
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = {
        "device": str(device),
        "model": checkpoint["model_name"],
        "best_epoch": checkpoint["epoch"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_loss": metrics["loss"],
        "metrics_path": str(args.metrics),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tiny-cnn", "resnet18"), default="tiny-cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best.pt"))
    parser.add_argument(
        "--metrics", type=Path, default=Path("artifacts/test_metrics.json")
    )
    parser.add_argument("--history", type=Path, default=Path("artifacts/history.json"))
    parser.add_argument(
        "--plot", type=Path, default=Path("artifacts/training_curves.png")
    )
    return parser


if __name__ == "__main__":
    train(parse_args().parse_args())
