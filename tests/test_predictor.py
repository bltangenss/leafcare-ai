"""Inference/validation regression tests using tiny temporary checkpoints."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch
from torch import nn
from PIL import Image

from app.predictor import ModelUnavailableError, Predictor


class TinyNet(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, count)

    def forward(self, value):
        return self.head(self.pool(self.conv(value)).flatten(1))


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(12)
    monkeypatch.setattr("app.predictor.build_model", lambda _name, count, pretrained=False: TinyNet(count))
    monkeypatch.setattr("app.predictor.gradcam_target_layer", lambda net, _name: net.conv)
    ck = {"model": "plantcnn", "classes": ["Apple___healthy", "Apple___Black_rot"],
          "img_size": 128, "state_dict": TinyNet(2).state_dict()}
    path = tmp_path / "plantcnn.pt"
    torch.save(ck, path)
    return tmp_path, path, ck


def test_inference_without_cam_has_no_hooks_or_gradients(checkpoint):
    pr = Predictor(checkpoint[0])
    result = pr.predict(Image.new("RGB", (20, 30)), cam=False)
    net = pr.models["plantcnn"]["net"]
    assert result["heatmap"] is None
    assert not net.conv._forward_hooks
    assert all(p.grad is None for p in net.parameters())
    assert sum(t["prob"] for t in result["top"]) == pytest.approx(1.)
    assert result["model_version"].startswith("sha256:")
    assert result["timings_ms"]["total"] >= result["timings_ms"]["inference"]


def test_cam_cleanup_after_failure_and_recovery(checkpoint, monkeypatch):
    pr = Predictor(checkpoint[0])
    original = pr._overlay
    def fail(*_args):
        raise RuntimeError("test overlay failure")
    monkeypatch.setattr(pr, "_overlay", fail)
    with pytest.raises(RuntimeError):
        pr.predict(Image.new("RGB", (10, 10)), cam=True)
    assert not pr.models["plantcnn"]["net"].conv._forward_hooks
    monkeypatch.setattr(pr, "_overlay", original)
    assert pr.predict(Image.new("RGB", (10, 10)), cam=True)["heatmap"].startswith("data:image/png;base64,")


def test_concurrent_gradcam_matches_serial_predictions(checkpoint):
    pr = Predictor(checkpoint[0])
    images = [Image.new("RGB", (20, 30), color) for color in ((30, 220, 40), (180, 10, 70))]
    expected = [pr.predict(image, cam=True) for image in images]
    barrier = threading.Barrier(2)
    def predict(image):
        barrier.wait()
        return pr.predict(image, cam=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(predict, images))
    for expected_result, actual_result in zip(expected, actual):
        assert actual_result["top"] == expected_result["top"]
        assert actual_result["heatmap"] == expected_result["heatmap"]
    assert not pr.models["plantcnn"]["net"].conv._forward_hooks


@pytest.mark.parametrize("change", [
    {"classes": ["Apple___healthy", "Apple___healthy"]},
    {"img_size": 224},
    {"model": "mobilenet"},
    {"class_to_idx": {"Apple___healthy": 1, "Apple___Black_rot": 0}},
    {"classes": ["Apple___healthy"]},
    {"preprocessing": {"color_mode": "BGR"}},
])
def test_invalid_checkpoint_does_not_prevent_service_start(checkpoint, change):
    folder, path, ck = checkpoint
    torch.save({**ck, **change}, path)
    pr = Predictor(folder)
    assert pr.models == {} and pr.unavailable["plantcnn"] == "checkpoint_invalid"
    with pytest.raises(ModelUnavailableError):
        pr.predict(Image.new("RGB", (20, 20)))


def test_individual_corrupt_checkpoint_does_not_hide_healthy_model(checkpoint):
    folder, _, _ = checkpoint
    (folder / "mobilenet.pt").write_bytes(b"not a checkpoint")
    pr = Predictor(folder)
    assert list(pr.models) == ["plantcnn"]
    assert pr.unavailable["mobilenet"] == "checkpoint_invalid"
    assert pr.default == "plantcnn"
    with pytest.raises(ModelUnavailableError):
        pr.predict(Image.new("RGB", (10, 10)), "mobilenet")
