"""Deterministic image manifests and split policies.

`legacy` reproduces the original file-level split for existing checkpoints.
`group_hash` is for NEW experiments and keeps connected source/hash groups together.
Neither policy establishes the absence of perceptually similar images.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "validation", "test")
SOURCE_ID = "huggingface:geraldmc/plantvillage-full; revision=unrecorded"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_samples(data_dir):
    root = Path(data_dir)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    samples = [(str(p), i) for i, name in enumerate(classes)
               for p in sorted((root / name).iterdir())
               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return classes, samples


def legacy_split(samples, seed=42):
    """Exact original 70/15/15 implementation (including per-class rounding)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for sample in samples:
        by[sample[1]].append(sample)
    result = [[], [], []]
    for values in by.values():
        rng.shuffle(values)
        a, b = int(len(values) * .70), int(len(values) * .85)
        for target, part in zip(result, (values[:a], values[a:b], values[b:])):
            target.extend(part)
    return tuple(result)


def group_hash_split(records, seed=42):
    """Assign whole connected groups; identical hashes never cross a split.

    Real group IDs can connect different images; hashes connect byte-identical
    images even across classes. Stratification is approximate for unequal groups.
    """
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        keys = [("hash", record["sha256"])]
        if record.get("source_group_id"):
            keys.append(("group", record["source_group_id"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    groups = sorted(groups.values(), key=lambda g: min(records[i]["relative_path"] for i in g))
    random.Random(seed).shuffle(groups)
    totals = Counter(r["class_label"] for r in records)
    counts = {name: Counter() for name in SPLITS}
    fractions = dict(zip(SPLITS, (.70, .15, .15)))
    for members in groups:
        labels = Counter(records[i]["class_label"] for i in members)
        # Largest remaining proportional deficit; seeded tie breaking comes
        # from the shuffled group order and stable split order.
        name = max(SPLITS, key=lambda s: sum(
            labels[c] * (fractions[s] * totals[c] - counts[s][c]) / max(1, totals[c])
            for c in labels))
        counts[name].update(labels)
        group_id = "component:" + hashlib.sha256("\n".join(
            sorted(records[i]["relative_path"] for i in members)).encode()).hexdigest()[:16]
        for i in members:
            records[i]["split"] = name
            records[i]["group_id"] = group_id
    return records


def make_manifest(data_dir, seed=42, policy="legacy", groups=None, limit=0):
    if policy not in {"legacy", "group_hash"}:
        raise ValueError(f"Unknown split policy: {policy}")
    root = Path(data_dir).resolve()
    source_path = root / "_source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    source_id = f"huggingface:{source['source_id']}; revision={source.get('revision', 'unrecorded')}" if source.get("source_id") else SOURCE_ID
    classes, samples = list_samples(root)
    if limit:
        samples = [s for i in range(len(classes)) for s in [s for s in samples if s[1] == i][:limit]]
    assigned = {str(Path(p).resolve()): name for name, part in zip(SPLITS, legacy_split(samples, seed)) for p, _ in part}
    groups = groups or {}
    records = []
    for path, index in samples:
        rel = Path(path).relative_to(root).as_posix()
        digest = sha256_file(path)
        source_group = groups.get(rel)
        records.append({"relative_path": rel, "class_label": classes[index], "class_index": index,
                        "source_id": source_id, "license": source.get("license", "unverified; confirm dataset card before redistribution"),
                        "source_group_id": source_group, "group_id": source_group or "sha256:" + digest,
                        "group_basis": "source_metadata" if source_group else "exact_bytes_only",
                        "sha256": digest, "split": assigned[str(Path(path).resolve())]})
    if policy == "group_hash":
        group_hash_split(records, seed)
    return classes, records


def manifest_text(records):
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in records)


def audit_manifest(records):
    hashes = defaultdict(list)
    groups = defaultdict(list)
    for record in records:
        hashes[record["sha256"]].append(record)
        if record.get("source_group_id"):
            groups[record["source_group_id"]].append(record)
    def crossings(values):
        return [{"key": key, "paths": [r["relative_path"] for r in rows],
                 "splits": sorted({r["split"] for r in rows})}
                for key, rows in values.items() if len({r["split"] for r in rows}) > 1]
    return {"files": len(records), "class_count": len({r["class_label"] for r in records}),
            "split_counts": dict(Counter(r["split"] for r in records)),
            "per_class": dict(Counter(r["class_label"] for r in records)),
            "duplicate_hash_groups": sum(len(v) > 1 for v in hashes.values()),
            "cross_split_exact_duplicates": crossings(hashes), "cross_split_source_groups": crossings(groups),
            "missing_source_group_ids": sum(not r.get("source_group_id") for r in records),
            "perceptual_similarity_check": "not_performed", "license_status": "unverified",
            "limitations": ["Historical checkpoint training-file membership was not hashed at training time.",
                            "File hashes identify exact duplicates only, not similar leaves or capture series.",
                            "No external field-image evaluation or calibration is established by this audit."]}


def save_manifest(path, classes, records, policy, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_text(records)
    path.write_text(payload, encoding="utf-8", newline="\n")
    summary = {"manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
               "split_policy": policy, "seed": seed, "classes": classes,
               "class_to_idx": {c: i for i, c in enumerate(classes)}, **audit_manifest(records)}
    path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
