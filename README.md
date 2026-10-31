# Vision Bench

Reproducible PyTorch image classification training, evaluation, checkpointing and inference. Vision Bench compares a small convolutional neural network with a CIFAR-adapted ResNet-18. It is designed to make the training process inspectable: data splitting, augmentation, optimization, checkpoint selection, and test evaluation are separate steps.

## Dataset and terms

The default experiment uses CIFAR-10 through `torchvision.datasets.CIFAR10`. The source page describes 60,000 32×32 color images in ten classes and asks users to cite the accompanying technical report. The page does not state a clear license grant, so this repository does not redistribute the dataset; review the source terms for your intended use and cite the report. The test partition is only evaluated after selecting the checkpoint by validation accuracy.

- [CIFAR-10 source and download](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR-10 technical report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [Torchvision CIFAR-10 loader](https://docs.pytorch.org/vision/0.18/generated/torchvision.datasets.CIFAR10.html)

## Setup

Python 3.10+ is supported. Install the appropriate PyTorch and torchvision builds for your platform first; CUDA use requires a CUDA-enabled PyTorch build and compatible NVIDIA driver. CPU is supported and is selected automatically when CUDA is unavailable.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Train and evaluate

```powershell
python -m vision_bench.train --model tiny-cnn --epochs 10 --device auto
python -m vision_bench.train --model resnet18 --epochs 10 --device auto
```

The first run downloads CIFAR-10 into `data/`. The training split is deterministic for a given seed; augmentation is applied only to training examples. Validation accuracy selects the best checkpoint. The official test partition is evaluated once after training. Outputs go to `artifacts/` by default:

- `best.pt`: model weights and class names from the best validation epoch
- `history.json` and `training_curves.png`: training/validation loss and accuracy
- `test_metrics.json`: test loss, accuracy, macro F1, confusion matrix, and per-class metrics

Useful options include `--seed`, `--batch-size`, `--learning-rate`, `--validation-fraction`, `--workers`, `--data-dir`, `--checkpoint`, and `--metrics`. Windows defaults to zero loader workers to avoid multiprocessing surprises.

## Image inference

```powershell
python -m vision_bench.inference path\to\image.jpg --checkpoint artifacts\best.pt --device auto
```

The input is converted to RGB, resized to 32×32, and normalized with the same channel statistics used by the data pipeline. The output is the predicted CIFAR-10 class and its softmax score; that score is not a calibrated probability.

## Tests

```powershell
python -m pytest -q
```

Tests use small in-memory image/tensor fixtures. They cover deterministic split behavior, disjoint train/validation indices, model output shapes, evaluation metrics, checkpoint round-trip, image inference, and CPU training. They do not download CIFAR-10 or establish model quality.

CI on Python 3.11 runs Ruff lint and formatting checks, the test suite on CPU, and a pip-audit scan of runtime dependencies.

## Limitations

This is a small educational benchmark, not a production classifier. CIFAR-10 images are low resolution and do not represent general image distributions. The ResNet-18 comparison is trained from random initialization; no pretrained weights are downloaded. No performance results are published in this README until a full dataset experiment has been run and reviewed. CUDA behavior is not verified on the current CPU-only PyTorch installation.

## License

The code is MIT licensed. The dataset is separately hosted and is not included in this repository; see the source page for its terms.
