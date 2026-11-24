"""PlantVillage датасетін (54 304 сурет, 38 класс) Hugging Face-тен Parquet түрінде жүктеп,
әр класстан PER_CLASS суреттен тұратын стратификацияланған ішкі жиынды data/plantvillage/ папкасына шығару.

Қолданылуы:  python scripts/download_dataset.py 250
"""
import argparse, io, json, os, random, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "geraldmc/plantvillage-full"
FILES = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data", "plantvillage")
CACHE = os.path.join(ROOT, "data", "_cache")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('per_class', type=int, nargs='?', default=250)
    parser.add_argument('--output', default=OUT, help='New or empty dataset directory')
    parser.add_argument('--revision', default='main', help='Hugging Face dataset commit/tag')
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error('per_class must be positive')
    target = os.path.abspath(args.output)
    if os.path.exists(target) and (not os.path.isdir(target) or os.listdir(target)):
        parser.error('Output already contains data. Choose a new --output directory to preserve existing experiments.')
    random.seed(42)
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(REPO, revision=args.revision).sha
    paths = [hf_hub_download(REPO, f, repo_type="dataset", revision=revision, local_dir=CACHE) for f in FILES]
    # 1) Тек класс белгілерін оқып, әр класстан кездейсоқ индекстерді таңдау
    labels = []
    for fi, p in enumerate(paths):
        col = pq.read_table(p, columns=["class_label"]).column("class_label").to_pylist()
        labels += [(fi, i, c) for i, c in enumerate(col)]
    by = {}
    for fi, i, c in labels:
        by.setdefault(c, []).append((fi, i))
    counts = {c: len(v) for c, v in sorted(by.items())}
    chosen = set()
    for c, lst in by.items():
        random.shuffle(lst)
        chosen.update(lst[:args.per_class])
    # 2) Суреттерді row group бойынша оқып, таңдалғандарын сақтау
    n = 0
    for fi, p in enumerate(paths):
        pf = pq.ParquetFile(p); offset = 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["image", "class_label"]).to_pylist()
            for j, row in enumerate(t):
                if (fi, offset + j) in chosen:
                    d = os.path.join(target, row["class_label"]); os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, f"{fi}_{offset + j:05d}.jpg"), "wb") as f:
                        f.write(row["image"]["bytes"])
                    n += 1
            offset += len(t)
        print("файл өңделді:", p, "| сақталды:", n, flush=True)
    with open(os.path.join(target, "_full_counts.json"), "w", encoding="utf8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=1)
    with open(os.path.join(target, "_source.json"), "w", encoding="utf8") as f:
        json.dump({"source_id": REPO, "revision": revision, "files": FILES,
                   "sampling_seed": 42, "per_class_limit": args.per_class,
                   "license": "unverified"}, f, ensure_ascii=False, indent=2)
    print("Толық датасет:", sum(counts.values()), "сурет,", len(counts), "класс. Ішкі жиын:", n)
    print("Dataset saved:", target, "| source revision:", revision)


if __name__ == "__main__":
    main()
