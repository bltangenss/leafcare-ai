"""Create a file-hashed dataset register without changing existing data/splits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.dataset import make_manifest, save_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "plantvillage")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "dataset_manifest_legacy.jsonl")
    parser.add_argument("--policy", choices=("legacy", "group_hash"), default="legacy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=Path, help="JSON map relative_path -> real source group ID")
    args = parser.parse_args()
    groups = json.loads(args.groups.read_text(encoding="utf-8")) if args.groups else None
    classes, records = make_manifest(args.data, args.seed, args.policy, groups)
    result = save_manifest(args.output, classes, records, args.policy, args.seed)
    print(json.dumps({k: result[k] for k in ("files", "class_count", "split_counts", "manifest_sha256",
                                             "duplicate_hash_groups", "missing_source_group_ids")}, indent=2))
    print(f"Cross-split exact duplicate groups: {len(result['cross_split_exact_duplicates'])}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
