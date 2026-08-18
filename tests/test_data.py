import numpy as np
import pandas as pd
import torch
from PIL import Image

from ocular import data
from ocular.channels import LABELS


def make_images(tmp_path, n_blink=3, n_other=7):
    rows = []
    for label, count in (("blink", n_blink), ("non-blink", n_other)):
        for i in range(count):
            path = tmp_path / f"{label}_{i}.png"
            Image.new("RGB", (256, 256), color=(i * 10, 100, 200)).save(path)
            rows.append({"path": str(path), "label": label})
    return pd.DataFrame(rows)


def test_dataset_returns_normalised_tensors(tmp_path):
    frame = make_images(tmp_path)
    dataset = data.TopoplotDataset(frame, train=False)

    image, target = dataset[0]
    assert image.shape == (3, data.INPUT_SIZE, data.INPUT_SIZE)
    assert image.dtype == torch.float32
    assert target in range(len(LABELS))


def test_labels_map_blink_to_one(tmp_path):
    frame = make_images(tmp_path)
    dataset = data.TopoplotDataset(frame, train=False)

    blink_rows = frame.index[frame["label"] == "blink"]
    assert all(dataset.targets[i] == 1 for i in blink_rows)


def test_training_transforms_differ_from_evaluation(tmp_path):
    """Augmentation should only be applied to the training split."""
    frame = make_images(tmp_path)
    train_first = data.TopoplotDataset(frame, train=True)[0][0]
    train_second = data.TopoplotDataset(frame, train=True)[0][0]
    eval_first = data.TopoplotDataset(frame, train=False)[0][0]
    eval_second = data.TopoplotDataset(frame, train=False)[0][0]

    assert torch.equal(eval_first, eval_second)
    assert not torch.equal(train_first, train_second)


def test_no_horizontal_flip_in_the_pipeline():
    """A mirrored scalp turns a left saccade into a right one."""
    names = [t.__class__.__name__ for t in data.build_transforms(train=True).transforms]
    assert not any("Flip" in name for name in names)


def test_balanced_sampler_evens_out_the_classes(tmp_path):
    frame = make_images(tmp_path, n_blink=2, n_other=98)
    dataset = data.TopoplotDataset(frame, train=True)
    sampler = data._balanced_sampler(dataset.targets)

    drawn = np.array([dataset.targets[i] for i in sampler])
    blink_share = (drawn == 1).mean()

    # Two blinks in a hundred, but they should come up around half the time
    assert 0.35 < blink_share < 0.65


def test_loader_keeps_row_order_when_not_training(tmp_path):
    """Benchmark pairing depends on evaluation order being stable."""
    frame = make_images(tmp_path)
    loader = data.make_loader(frame, train=False, batch_size=4, num_workers=0)

    targets = torch.cat([t for _, t in loader]).numpy()
    expected = data.TopoplotDataset(frame).targets
    assert np.array_equal(targets, expected)
