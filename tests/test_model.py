import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from vision_bench.engine import evaluate, save_training_artifacts, train_model
from vision_bench.inference import load_checkpoint, predict_image
from vision_bench.models import TinyCNN, build_model


def test_tiny_cnn_output_shape():
    assert TinyCNN()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_resnet18_is_adapted_for_small_rgb_images():
    model = build_model("resnet18", num_classes=4)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 4)


def test_evaluation_returns_confusion_and_class_metrics():
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    metrics = evaluate(
        TinyCNN(2),
        DataLoader(TensorDataset(images, labels), batch_size=2),
        "cpu",
        ["a", "b"],
    )
    assert len(metrics["confusion_matrix"]) == 2
    assert set(metrics["per_class"]) >= {"a", "b", "macro avg"}
    assert 0 <= metrics["accuracy"] <= 1


def test_training_saves_checkpoint_and_image_inference(tmp_path):
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    checkpoint_path = tmp_path / "best.pt"
    history = train_model(
        loader,
        loader,
        model_name="tiny-cnn",
        class_names=["a", "b"],
        epochs=1,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
    )
    assert len(history) == 1
    history_path, plot_path = tmp_path / "history.json", tmp_path / "curves.png"
    save_training_artifacts(history, history_path, plot_path)
    assert history_path.is_file() and plot_path.is_file()
    model, classes = load_checkpoint(checkpoint_path, torch.device("cpu"))
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color="red").save(image_path)
    prediction = predict_image(model, classes, image_path, torch.device("cpu"))
    assert prediction["class"] in classes
    assert 0 <= prediction["confidence"] <= 1


def test_unknown_model_name_is_rejected():
    try:
        build_model("unknown")
    except ValueError as error:
        assert "Unknown model" in str(error)
    else:
        raise AssertionError("unknown model name should fail")
