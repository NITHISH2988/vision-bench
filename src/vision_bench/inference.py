"""Load a saved classifier and predict one image."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .data import MEAN, STD
from .models import build_model


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_name"], len(checkpoint["class_names"])).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["class_names"]


def predict_image(model, class_names: list[str], image_path: Path, device) -> dict:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    with Image.open(image_path) as image:
        tensor = (
            transform(image.convert("RGB").resize((32, 32))).unsqueeze(0).to(device)
        )
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    index = int(probabilities.argmax())
    return {"class": class_names[index], "confidence": float(probabilities[index])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch installation cannot access CUDA"
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model, classes = load_checkpoint(args.checkpoint, device)
    print(predict_image(model, classes, args.image, device))


if __name__ == "__main__":
    main()
