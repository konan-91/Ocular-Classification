import pytest
import torch

from ocular import model as model_module
from ocular.channels import LABELS


def tiny_model():
    return model_module.build("resnet18", pretrained=False)


def test_head_matches_the_label_count():
    assert tiny_model().fc.out_features == len(LABELS)


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown architecture"):
        model_module.build("resnet99", pretrained=False)


def test_freezing_leaves_only_the_head_trainable():
    model = tiny_model()
    model_module.set_backbone_trainable(model, False)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}


def test_unfreezing_restores_every_parameter():
    model = tiny_model()
    model_module.set_backbone_trainable(model, False)
    model_module.set_backbone_trainable(model, True)

    assert all(p.requires_grad for p in model.parameters())


def test_save_and_load_preserve_the_weights(tmp_path):
    model = tiny_model()
    path = tmp_path / "model.pt"
    model_module.save(model, path, {"architecture": "resnet18", "note": "test"})

    loaded, metadata = model_module.load(path, device=torch.device("cpu"))

    assert metadata["note"] == "test"
    for original, restored in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(original.cpu(), restored.cpu())


def test_forward_pass_returns_one_logit_per_class():
    model = tiny_model().eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, len(LABELS))
