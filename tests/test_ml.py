"""Small deterministic reproducibility tests; never touch the production data."""
import ast
from pathlib import Path

from ml.dataset import audit_manifest, group_hash_split, legacy_split, make_manifest, save_manifest, sha256_file


def test_legacy_split_is_original_order_and_rounding():
    import random
    samples = [(f"{label}-{i}", label) for label in range(3) for i in range(11)]
    expected = [[], [], []]
    rng = random.Random(42)
    for label in range(3):
        rows = [s for s in samples if s[1] == label]
        rng.shuffle(rows)
        for target, part in zip(expected, (rows[:7], rows[7:9], rows[9:])):
            target.extend(part)
    assert legacy_split(samples) == tuple(expected)
    assert len(set(p for rows in legacy_split(samples) for p, _ in rows)) == len(samples)


def test_manifest_hash_repeatable_and_metadata_preserved(tmp_path):
    import json
    for label in ("a", "b"):
        (tmp_path / label).mkdir()
        for i in range(10):
            (tmp_path / label / f"{i:02}.jpg").write_bytes(f"{label}-{i}".encode())
        (tmp_path / label / "notes.txt").write_text("not an image")
    (tmp_path / "_source.json").write_text(json.dumps({"source_id":"example/dataset", "revision":"abc123", "license":"unverified"}))
    classes, records = make_manifest(tmp_path)
    assert classes == ["a", "b"]
    assert len(records) == 20
    assert records[0]["source_id"] == "huggingface:example/dataset; revision=abc123"
    assert records[0]["license"] == "unverified"
    assert records[0]["group_basis"] == "exact_bytes_only"
    assert records == make_manifest(tmp_path)[1]
    output = tmp_path / "manifest.jsonl"
    summary = save_manifest(output, classes, records, "legacy", 42)
    assert summary["manifest_sha256"] == sha256_file(output)
    assert summary["missing_source_group_ids"] == 20
    assert summary["perceptual_similarity_check"] == "not_performed"


def test_group_and_hash_connections_never_leak():
    records = [dict(relative_path=f"a/{i}.jpg", class_label="a", sha256=str(i), source_group_id=None) for i in range(20)]
    # A real group joins 0+1; an identical hash joins 1+2. All three must stay together.
    records[0]["source_group_id"] = records[1]["source_group_id"] = "leaf-1"
    records[2]["sha256"] = records[1]["sha256"]
    grouped = group_hash_split(records, seed=42)
    assert len({r["split"] for r in grouped[:3]}) == 1
    assert len({r["group_id"] for r in grouped[:3]}) == 1
    assert {r["split"] for r in grouped} == {"train", "validation", "test"}
    audit = audit_manifest(grouped)
    assert not audit["cross_split_exact_duplicates"]
    assert not audit["cross_split_source_groups"]


def test_audit_exposes_cross_split_duplicates():
    rows = [dict(relative_path="a/1.jpg", class_label="a", sha256="same", split="train"),
            dict(relative_path="a/2.jpg", class_label="a", sha256="same", split="test")]
    report = audit_manifest(rows)
    assert report["duplicate_hash_groups"] == 1
    assert len(report["cross_split_exact_duplicates"]) == 1
    assert report["cross_split_exact_duplicates"][0]["splits"] == ["test", "train"]


def test_new_baseline_models_have_correct_output_without_changing_legacy():
    import torch
    from ml.models import build_model, count_params
    torch.set_num_threads(2)
    for name in ("baseline", "regularized", "plantcnn"):
        model = build_model(name, 5).eval()
        with torch.inference_mode():
            assert model(torch.zeros(2, 3, 128, 128)).shape == (2, 5)
    assert count_params(build_model("regularized", 5)) > count_params(build_model("baseline", 5))


def test_plot_module_is_syntactically_valid_and_skips_missing_curves():
    path = Path(__file__).resolve().parents[1] / "ml" / "plots.py"
    ast.parse(path.read_text(encoding="utf-8"))
    from ml.plots import curves
    assert curves({"history": None}) is None


def test_phone_photo_training_tensor_matches_serving_tensor(tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    from ml.train import load_tensor, normalize
    from app.predictor import Predictor

    pixels = np.zeros((30, 60, 3), dtype=np.uint8)
    pixels[:, :20] = (230, 30, 40)
    pixels[10:, 20:] = (10, 200, 70)
    photo = Image.fromarray(pixels)
    metadata = Image.Exif()
    metadata[274] = 6  # Phone orientation: rotate 90 degrees clockwise.
    path = tmp_path / "phone.jpg"
    photo.save(path, exif=metadata)
    training, _ = load_tensor([(str(path), 0)], 128)
    with Image.open(path) as source:
        serving, _ = Predictor.__new__(Predictor).preprocess(source, 128)
    torch.testing.assert_close(normalize(training), serving, rtol=0, atol=0)
