"""Small CNN baseline and a CIFAR-adapted ResNet-18 comparison model."""

from torch import nn
from torchvision.models import resnet18


class TinyCNN(nn.Module):
    """A compact two-stage CNN baseline for 32x32 RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, images):
        return self.head(self.features(images).flatten(1))


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Build a model; ResNet-18 starts from random weights for reproducibility."""
    if name == "tiny-cnn":
        return TinyCNN(num_classes)
    if name == "resnet18":
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model {name!r}; choose 'tiny-cnn' or 'resnet18'")
