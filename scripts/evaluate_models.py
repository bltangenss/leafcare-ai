"""Re-evaluate existing checkpoints without retraining or replacing old results.

The legacy split is reconstructed from current sorted files and seed 42. Its
counts/classes must match reports/split.json; original membership was not hashed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from ml.dataset import make_manifest, save_manifest, sha256_file
from ml.models import build_model, count_params, MEAN, STD, MODEL_SPECS


def preprocess(path, size):
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    return (pixels - MEAN) / STD


class Images(Dataset):
    def __init__(self, root, rows, size):
        self.root, self.rows, self.size = root, rows, size
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return preprocess(self.root / row["relative_path"], self.size), row["class_index"]


def percentiles(values):
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "mean": float(np.mean(values)), "samples": len(values)}


def evaluate_checkpoint(name, path, data, rows, classes, manifest, batch_size, latency_samples):
    started = time.perf_counter()
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("classes") != classes or ck.get("model") != name:
        raise ValueError(f"Checkpoint class mapping/architecture mismatch: {path}")
    if ck.get("dataset_manifest_sha256") and ck["dataset_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Checkpoint was trained with a different manifest; use that run's test split and metrics.")
    if ck.get("preprocessing", {}).get("exif_orientation") == "transpose":
        raise ValueError("This command reconstructs legacy preprocessing only; use the new experiment's recorded evaluation.")
    size = ck["img_size"]
    net = build_model(name, len(classes), pretrained=False).eval().to(memory_format=torch.channels_last)
    net.load_state_dict(ck["state_dict"], strict=True)
    loader = DataLoader(Images(data, rows, size), batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, truths, probabilities = [], [], []
    with torch.inference_mode():
        for xb, yb in loader:
            prob = net(xb.contiguous(memory_format=torch.channels_last)).softmax(1)
            predictions.extend(prob.argmax(1).tolist())
            truths.extend(yb.tolist())
            probabilities.extend(prob.tolist())
    probabilities = np.asarray(probabilities)
    labels = list(range(len(classes)))
    cm = confusion_matrix(truths, predictions, labels=labels)
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    pp, forward, total = [], [], []
    latency_rows = rows[::max(1, len(rows) // latency_samples)][:latency_samples]
    with torch.inference_mode():
        warm = preprocess(data / latency_rows[0]["relative_path"], size).unsqueeze(0).contiguous(memory_format=torch.channels_last)
        for _ in range(5):
            net(warm)
        for row in latency_rows:
            start = time.perf_counter()
            x = preprocess(data / row["relative_path"], size).unsqueeze(0).contiguous(memory_format=torch.channels_last)
            before = time.perf_counter()
            net(x)
            finish = time.perf_counter()
            pp.append((before-start)*1000)
            forward.append((finish-before)*1000)
            total.append((finish-start)*1000)
    return {"schema_version": 1, "kind": "checkpoint_reevaluation", "model": name,
            "title": MODEL_SPECS[name]["title"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": sha256_file(path), "dataset_manifest_sha256": manifest["manifest_sha256"],
            "split_policy": "legacy", "split_seed": 42, "split_reconstruction": "current sorted files; original membership hash unavailable",
            "classes": classes, "class_to_idx": {c:i for i,c in enumerate(classes)},
            "img_size": size, "preprocessing": {"color_mode": "RGB", "resize": "bilinear_square", "exif_orientation": "legacy_ignored", "mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist()},
            "params": count_params(net), "model_size_mb": path.stat().st_size / 1e6, "test_samples": len(rows),
            "test_acc": float(np.mean(np.array(truths) == predictions)),
            "test_top3": float(np.mean([y in np.argsort(-p)[:3] for y,p in zip(truths, probabilities)])),
            "precision_macro": float(precision_score(truths, predictions, labels=labels, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(truths, predictions, labels=labels, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(truths, predictions, labels=labels, average="macro", zero_division=0)),
            "report": classification_report(truths, predictions, labels=labels, target_names=classes, output_dict=True, zero_division=0),
            "confusion": cm.tolist(), "confusion_normalized": normalized.tolist(),
            "latency_ms": {"preprocessing": percentiles(pp), "forward": percentiles(forward), "total": percentiles(total),
                           "batch_size": 1, "warmup_calls": 5, "gradcam_included": False, "http_included": False},
            "infer_ms_cpu": float(np.mean(forward)),
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "torchvision": str(torchvision.__version__), "platform": platform.platform(),
                            "processor": platform.processor(), "device": "cpu", "threads": torch.get_num_threads()},
            "duration_s": time.perf_counter() - started, "history": None,
            "training_history_status": "not_reconstructed_from_checkpoint", "calibrated": False,
            "external_evaluation": "not_performed", "perceptual_duplicate_audit": "not_performed",
            "cross_split_exact_duplicate_groups": len(manifest["cross_split_exact_duplicates"]),
            "limitations": manifest["limitations"]}


def save_confusion(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, key, title in zip(axes, ("confusion", "confusion_normalized"), ("Counts", "Row-normalized")):
        im = ax.imshow(result[key], cmap="Greens")
        ax.set(title=title, xlabel="Predicted class index", ylabel="True class index")
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle(f"{result['title']} | n={result['test_samples']} | legacy split")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "plantvillage")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "evaluation_current")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=["plantcnn", "mobilenet"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--latency-samples", type=int, default=50)
    args = parser.parse_args()
    if min(args.batch_size, args.threads, args.latency_samples) < 1:
        parser.error("batch-size, threads and latency-samples must be positive")
    torch.set_num_threads(args.threads)
    classes, records = make_manifest(args.data, 42, "legacy")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = save_manifest(args.output / "dataset_manifest.jsonl", classes, records, "legacy", 42)
    saved_split = json.loads((ROOT / "reports" / "split.json").read_text(encoding="utf-8"))
    expected = {"train": saved_split["train"], "validation": saved_split["val"], "test": saved_split["test"]}
    if classes != saved_split["classes"] or manifest["split_counts"] != expected or manifest["per_class"] != saved_split["per_class"]:
        raise ValueError("Current data does not match saved legacy counts/classes; evaluation stopped")
    rows = [r for r in records if r["split"] == "test"]
    print(f"Manifest {manifest['manifest_sha256']} | {len(rows)} test images", flush=True)
    for name in args.models:
        result = evaluate_checkpoint(name, ROOT / "models" / f"{name}.pt", args.data, rows, classes,
                                     manifest, args.batch_size, args.latency_samples)
        (args.output / f"{name}_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        save_confusion(result, args.output / f"{name}_confusion.png")
        print(f"{name}: accuracy={result['test_acc']:.6f}, macro_f1={result['f1_macro']:.6f}, duration={result['duration_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
