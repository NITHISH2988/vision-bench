# Developer Guide: Vision Bench

Vision Bench makes image-classification training inspectable. It compares a compact CNN with a CIFAR-adapted ResNet-18, both initialized from random weights.

## Architecture and data flow

```text
train CLI → set_seed + choose device → make_loaders
                                      ├─ augmented train subset
                                      ├─ clean validation subset
                                      └─ official test partition
                 train_model → epoch train/evaluate → best validation checkpoint
                                              ↓ reload best checkpoint
                                    evaluate test once → JSON + plots
inference CLI → load_checkpoint → same normalization → logits → class/score
```

`data.make_loaders()` creates two views of the CIFAR-10 training data: augmentation is attached only to the training view; validation uses the same indices with deterministic transforms. A seeded permutation makes disjoint train and validation subsets. Test data remains separate. `models.TinyCNN` uses two convolution/ReLU/pooling stages and adaptive average pooling. `build_model("resnet18")` adapts the first convolution and removes the initial max-pool for 32×32 inputs, then replaces the classifier head.

`engine.set_seed()` seeds Python, NumPy, and PyTorch. `_run_batches()` owns the common batch loop: training enables gradients, clears gradients, calculates cross-entropy, backpropagates, and steps AdamW; evaluation runs without gradients. `train_model()` evaluates each epoch and atomically replaces the checkpoint only when validation accuracy improves. The checkpoint stores the architecture name, class names, epoch, validation accuracy, and state dict. `train.train()` reloads that best checkpoint before measuring the test partition. The test report contains accuracy, macro F1, confusion matrix, and per-class metrics.

## Directory map and entry points

- `src/vision_bench/data.py`: normalization constants and `make_loaders`.
- `src/vision_bench/models.py`: `TinyCNN` and `build_model`.
- `src/vision_bench/engine.py`: `set_seed`, `_run_batches`, `evaluate`, `train_model`, and `save_training_artifacts`.
- `src/vision_bench/train.py`: device selection, CLI argument parsing, and complete train/evaluate orchestration; run `python -m vision_bench.train`.
- `src/vision_bench/inference.py`: `load_checkpoint`, `predict_image`, and single-image CLI.
- `tests/`: model shapes, fixture-based data splits, CPU training/checkpointing, metrics, and inference behavior.
- `.github/workflows/ci.yml`: CPU-only tests, Ruff, and dependency audit.

## Reproduce training and inference

Install PyTorch and torchvision builds compatible with your machine first; this matters for CUDA. Then install the project in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m vision_bench.train --model tiny-cnn --epochs 1 --device cpu --batch-size 128
python -m pytest -q
```

The first training run downloads CIFAR-10 into `data/`. `--epochs 1` is a smoke run, not a meaningful benchmark. A complete run can use `--epochs 10 --device auto`; compare both model names with the same seed and settings. The run produces `artifacts/best.pt`, `history.json`, `training_curves.png`, and `test_metrics.json`. No model-quality result should be claimed until a full experiment has run and its conditions and metrics have been recorded.

For inference, provide a local image and checkpoint:

```powershell
python -m vision_bench.inference path\to\image.jpg --checkpoint artifacts\best.pt --device auto
```

The image is converted to RGB, resized to 32×32, and normalized using CIFAR-10 channel statistics. Output includes a class and softmax score; it is not calibrated confidence. `--device cpu` is the reliable fallback. `--device cuda` fails clearly if CUDA is unavailable; `auto` selects CUDA only when the installed PyTorch can use it.

## Concepts to be ready to explain

1. **Convolution and receptive fields:** learned kernels detect local patterns; stacking layers builds larger effective receptive fields.
2. **Pooling and adaptive pooling:** spatial downsampling reduces computation; adaptive pooling gives a fixed-size feature vector.
3. **Residual connections:** ResNet blocks learn residual updates, helping optimization of deeper networks; this project uses torchvision’s standard ResNet-18 blocks.
4. **Backpropagation:** cross-entropy gradients flow through logits to update parameters with AdamW.
5. **Train/eval mode:** `model.train()` activates training behavior, while `model.eval()` and inference/no-grad contexts change evaluation behavior and avoid gradient storage.
6. **Augmentation:** random crop/flip are applied only to training examples to expose varied views without contaminating validation.
7. **Checkpoint selection:** validation accuracy picks a checkpoint; the test partition is held back for final evaluation.
8. **Macro F1 and confusion matrix:** class-wise performance matters beyond aggregate accuracy.
9. **Reproducibility limits:** seeds control key RNGs, but hardware kernels and library/device differences can still make bitwise results vary.
10. **Transfer learning trade-off:** ResNet here starts from random weights, so it demonstrates architecture comparison, not pretrained transfer learning.

## Tests, CI, and extension points

The tests use small in-memory fixtures and exercise CPU behavior without downloading CIFAR-10. Run `python -m pytest -q`, `python -m ruff check .`, and `python -m ruff format --check .`. GitHub Actions runs the same suite on CPU plus dependency auditing. CUDA is optional and is not proven by the CPU CI result.

Useful extensions are a benchmark manifest (hardware, versions, seed, settings), multiple-seed mean/variance comparisons, and an optional pretrained experiment with explicit weight provenance and a fair fine-tuning protocol. Keep the official test partition out of model selection.

## Current limits and troubleshooting

- CIFAR-10’s source requests citation but does not present a clear dataset license; data is downloaded separately and not redistributed.
- No full dataset experiment or CUDA run is claimed here.
- If torchvision fails to import, install a torch/torchvision pair built for the same CPU/CUDA runtime.
- If a checkpoint is missing, train first or pass the correct `--checkpoint` path.
- On Windows, `--workers 0` avoids multiprocessing setup problems and is already the CLI default.
- `weights_only=True` limits checkpoint deserialization behavior; use checkpoints from trusted sources anyway.

