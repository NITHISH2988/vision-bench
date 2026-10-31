from typing import ClassVar

from PIL import Image

from vision_bench.data import make_loaders


class FakeCifar10:
    classes: ClassVar[list[str]] = [f"class-{index}" for index in range(10)]

    def __init__(self, root, train, download, transform):
        self.length = 20 if train else 10
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = Image.new("RGB", (32, 32), color=(index, index, index))
        return self.transform(image), index % 10


def test_split_is_disjoint_reproducible_and_test_is_held_out(tmp_path, monkeypatch):
    monkeypatch.setattr("vision_bench.data.datasets.CIFAR10", FakeCifar10)
    loaders = make_loaders(tmp_path, batch_size=4, validation_fraction=0.25, seed=9)
    repeated = make_loaders(tmp_path, batch_size=4, validation_fraction=0.25, seed=9)
    train_indices = set(loaders[0].dataset.indices)
    validation_indices = set(loaders[1].dataset.indices)

    assert len(train_indices) == 15
    assert len(validation_indices) == 5
    assert train_indices.isdisjoint(validation_indices)
    assert loaders[0].dataset.indices == repeated[0].dataset.indices
    assert len(loaders[2].dataset) == 10


def test_invalid_validation_fraction_fails(tmp_path):
    try:
        make_loaders(tmp_path, validation_fraction=1.0)
    except ValueError as error:
        assert "validation_fraction" in str(error)
    else:
        raise AssertionError("invalid validation split should fail")
