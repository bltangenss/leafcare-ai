"""Validated CPU model loading and serialized, optional Grad-CAM inference."""
from __future__ import annotations
import base64
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import threading
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from matplotlib import colormaps
from app.disease_info import INFO
from ml.models import build_model, gradcam_target_layer, MODEL_SPECS, MEAN, STD

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LOGGER = logging.getLogger("leafcare.models")


class ModelUnavailableError(ValueError):
    """Requested model is not available; never silently substitute a model."""


class Predictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models = {}
        self.unavailable = {}
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        for name in MODEL_SPECS:
            path = Path(models_dir) / f"{name}.pt"
            if not path.is_file():
                if name in ("plantcnn", "mobilenet"):
                    self.unavailable[name] = "checkpoint_missing"
                continue
            try:
                self.models[name] = self._load(name, path)
            except Exception as exc:
                self.unavailable[name] = "checkpoint_invalid"
                LOGGER.warning(json.dumps({"event": "model_load_failed", "model": name,
                                           "error_type": type(exc).__name__}))

    @staticmethod
    def _load(name, path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ck, dict) or ck.get("model") != name:
            raise ValueError("Checkpoint architecture does not match its model identifier")
        classes = ck.get("classes")
        if (not isinstance(classes, list) or not classes
                or not all(isinstance(c, str) and c in INFO for c in classes)
                or len(set(classes)) != len(classes)):
            raise ValueError("Invalid ordered class mapping")
        if "class_to_idx" in ck and ck["class_to_idx"] != {c: i for i, c in enumerate(classes)}:
            raise ValueError("Conflicting class mapping")
        size = ck.get("img_size")
        if type(size) is not int or size != MODEL_SPECS[name]["img_size"]:
            raise ValueError("Input size differs from the trained architecture configuration")
        preprocessing = {"color_mode": "RGB", "resize": [size, size], "interpolation": "bilinear", "exif_orientation": "transpose",
                         "scale": 255.0, "mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist()}
        declared = ck.get("preprocessing")
        if declared is not None:
            if declared.get("exif_orientation", "transpose") != "transpose":
                raise ValueError("EXIF orientation policy mismatch")
            for key in ("color_mode", "resize", "interpolation", "scale"):
                if declared.get(key) != preprocessing[key]:
                    raise ValueError("Preprocessing configuration mismatch")
            for key in ("mean", "std"):
                if not np.allclose(declared.get(key, []), preprocessing[key], atol=1e-6):
                    raise ValueError("Normalization mismatch")
        net = build_model(name, len(classes), pretrained=False)
        net.load_state_dict(ck["state_dict"], strict=True)
        net.eval()
        with torch.inference_mode():
            output = net(torch.zeros(1, 3, size, size))
        if output.shape != (1, len(classes)) or not torch.isfinite(output).all():
            raise ValueError("Invalid model output shape or values")
        return {"net": net, "classes": classes, "size": size, "title": MODEL_SPECS[name]["title"],
                "version": f"sha256:{digest}", "checkpoint_sha256": digest,
                "preprocessing": preprocessing,
                "preprocessing_provenance": "checkpoint" if declared else "legacy_repository_config",
                "params": sum(p.numel() for p in net.parameters()), "lock": threading.Lock()}

    @property
    def default(self):
        return "mobilenet" if "mobilenet" in self.models else next(iter(self.models), None)

    def preprocess(self, img: Image.Image, size: int):
        img = ImageOps.exif_transpose(img).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        return ((x - MEAN) / STD).unsqueeze(0), img

    def predict(self, img: Image.Image, model_name: str | None = None, topk: int = 5, cam: bool = True):
        name = model_name if model_name else self.default
        if name not in self.models:
            raise ModelUnavailableError("Сұралған модель қолжетімсіз. Қолжетімді модельді таңдаңыз.")
        if not isinstance(topk, int) or topk < 1:
            raise ValueError("topk must be a positive integer")
        total_start = time.perf_counter()
        spec = self.models[name]
        net = spec["net"]
        pre_start = time.perf_counter()
        x, resized = self.preprocess(img, spec["size"])
        preprocess_ms = (time.perf_counter() - pre_start) * 1000
        heatmap = None
        gradcam_ms = 0.0
        wait_start = time.perf_counter()
        # Hooks belong to the shared model; serialize inference in both modes.
        with spec["lock"]:
            queue_ms = (time.perf_counter() - wait_start) * 1000
            handle = None
            acts = {}
            try:
                if cam:
                    def capture(_module, _inputs, output):
                        acts["activation"] = output
                    handle = gradcam_target_layer(net, name).register_forward_hook(capture)
                start = time.perf_counter()
                if cam:
                    with torch.enable_grad():
                        logits = net(x.requires_grad_(True))
                else:
                    with torch.inference_mode():
                        logits = net(x)
                forward_ms = (time.perf_counter() - start) * 1000
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Model returned non-finite values")
                probs = logits.detach().softmax(1)[0]
                top = torch.topk(probs, k=min(topk, len(probs)))
                if cam:
                    start = time.perf_counter()
                    activation = acts["activation"]
                    with torch.enable_grad():
                        gradient = torch.autograd.grad(logits[0, top.indices[0]], activation)[0]
                    a, g = activation.detach(), gradient.detach()
                    cam_map = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
                    cam_map = F.interpolate(cam_map, size=resized.size[::-1], mode="bilinear", align_corners=False)[0, 0]
                    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
                    heatmap = self._overlay(resized, cam_map.numpy())
                    gradcam_ms = (time.perf_counter() - start) * 1000
            finally:
                if handle is not None:
                    handle.remove()
        total_ms = (time.perf_counter() - total_start) * 1000
        return {"model": name, "model_title": spec["title"], "model_version": spec["version"],
                "time_ms": round(total_ms, 2),
                "timings_ms": {"preprocess": round(preprocess_ms, 2), "queue": round(queue_ms, 2),
                               "inference": round(forward_ms, 2), "gradcam": round(gradcam_ms, 2), "total": round(total_ms, 2)},
                "top": [{"cls": spec["classes"][i], "prob": float(p), "index": i}
                        for p, i in zip(top.values.tolist(), top.indices.tolist())], "heatmap": heatmap}

    @staticmethod
    def _overlay(img: Image.Image, cam: np.ndarray, alpha=0.45):
        color = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
        out = (np.asarray(img).astype(np.float32) * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(out).resize((320, 320), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
