"""CNN модельдерін PlantVillage ішкі жиынында оқыту.

Қолданылуы:
    python -m ml.train --model plantcnn --epochs 20
    python -m ml.train --model mobilenet --epochs 8
"""
import argparse, csv, json, os, platform, random, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2 as T

from ml.models import build_model, count_params, MODEL_SPECS, MEAN, STD
from ml.dataset import list_samples as scan_samples, legacy_split, make_manifest, save_manifest, sha256_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "plantvillage")
OUT = os.path.join(ROOT, "models")
REP = os.path.join(ROOT, "reports")


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def list_samples():
    return scan_samples(DATA)


def split(samples, seed=42):
    """Legacy split retained solely for compatibility with existing weights."""
    return legacy_split(samples, seed)


def load_tensor(samples, size):
    """Суреттерді жадқа uint8 тензор ретінде алдын ала жүктеу (CPU-да жылдамдық үшін)."""
    x = torch.empty((len(samples), 3, size, size), dtype=torch.uint8)
    y = torch.tensor([s[1] for s in samples], dtype=torch.long)
    for i, (p, _) in enumerate(samples):
        with Image.open(p) as source:
            im = ImageOps.exif_transpose(source).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        x[i] = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
    return x, y


def normalize(xb):
    return ((xb.float() / 255.0 - MEAN) / STD).contiguous(memory_format=torch.channels_last)


AUG = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(25)], p=0.5),
    T.ColorJitter(0.25, 0.25, 0.25, 0.03),
])


def augment(xb, size):
    """Деректерді кеңейту: аудару, бұру, кесу, түс өзгерту."""
    crop = T.RandomResizedCrop(size, scale=(0.75, 1.0), antialias=True)
    return torch.stack([crop(AUG(img)) for img in xb])


class Head(nn.Module):
    """MobileNetV2-нің мұздатылмаған жоғарғы бөлігі: features[cut:] -> GAP -> классификатор."""
    def __init__(self, m, cut):
        super().__init__()
        self.top, self.classifier = m.features[cut:], m.classifier
    def forward(self, f):
        return self.classifier(torch.flatten(nn.functional.adaptive_avg_pool2d(self.top(f), 1), 1))


def evaluate(model, loader, prep=None):
    prep = prep or normalize
    model.eval(); crit = nn.CrossEntropyLoss()
    tot, correct, loss_sum = 0, 0, 0.0
    preds, gts, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(prep(xb))
            loss_sum += crit(out, yb).item() * len(yb)
            p = out.softmax(1)
            pr = p.argmax(1)
            correct += (pr == yb).sum().item(); tot += len(yb)
            preds.append(pr); gts.append(yb); probs.append(p)
    return loss_sum / tot, correct / tot, torch.cat(preds).numpy(), torch.cat(gts).numpy(), torch.cat(probs).numpy()


def main():
    if hasattr(os.sys.stdout, "reconfigure"):
        os.sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="plantcnn", choices=list(MODEL_SPECS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0, help="әр класстан ең көп сурет саны (тексеру үшін)")
    ap.add_argument("--out-suffix", default="", help="нәтиже файлдарына жұрнақ")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--cut", type=int, default=14, help="MobileNet: мұздатылатын блоктар саны")
    ap.add_argument("--views", type=int, default=2, help="MobileNet: кэштелетін аугментация нұсқалары")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--split-policy", choices=("group_hash", "legacy"), default="group_hash")
    ap.add_argument("--groups", type=Path, help="JSON map relative_path -> actual source group ID")
    ap.add_argument("--run-id", help="Unique artifact directory name; defaults to UTC timestamp")
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation Macro F1; 0 disables")
    ap.add_argument("--augmentation", choices=("on", "off"), default="on")
    args = ap.parse_args()
    if min(args.epochs, args.bs, args.views, args.threads) < 1 or args.limit < 0 or args.patience < 0:
        ap.error("epochs, bs, views and threads must be positive; limit/patience cannot be negative")
    if not 1 <= args.cut <= 18:
        ap.error("cut must be between 1 and 18")
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    size = MODEL_SPECS[args.model]["img_size"]
    os.makedirs(OUT, exist_ok=True); os.makedirs(REP, exist_ok=True)
    run_id = args.run_id or f"{args.model}{args.out_suffix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in run_id):
        ap.error("run-id may contain only letters, digits, underscores and hyphens")
    run_dir = Path(REP) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "checkpoint.pt"
    groups = json.loads(args.groups.read_text(encoding="utf-8")) if args.groups else None
    classes, records = make_manifest(DATA, args.seed, args.split_policy, groups, args.limit)
    manifest = save_manifest(run_dir / "dataset_manifest.jsonl", classes, records, args.split_policy, args.seed)
    parts = {name: [(str(Path(DATA) / r["relative_path"]), r["class_index"]) for r in records if r["split"] == name]
             for name in ("train", "validation", "test")}
    tr, va, te = (parts[name] for name in ("train", "validation", "test"))
    if any(not part for part in (tr, va, te)):
        ap.error("Dataset must provide a nonempty train, validation and test split")
    print(f"Кластар: {len(classes)} | train {len(tr)} / val {len(va)} / test {len(te)}")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    import torchvision
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "torchvision": str(torchvision.__version__), "platform": platform.platform(),
                   "device": "cpu", "threads": args.threads, "git_commit": commit,
                   "source_sha256": {name: sha256_file(Path(ROOT) / name) for name in ("ml/train.py", "ml/models.py", "ml/dataset.py")}}
    config = {**vars(args), "groups": str(args.groups) if args.groups else None, "run_id": run_id,
              "purpose": "pipeline_smoke" if args.limit else "training_experiment",
              "dataset_manifest_sha256": manifest["manifest_sha256"], "class_to_idx": {c:i for i,c in enumerate(classes)},
              "image_size": size, "normalization": {"mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist()},
              "preprocessing": {"color_mode": "RGB", "resize": [size, size], "interpolation": "bilinear", "exif_orientation": "transpose",
                                "scale": 255.0, "mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist()},
              "resize": "RGB_bilinear_square", "selection_criterion": "validation_macro_f1",
              "optimizer": "AdamW", "lr": args.lr or (1e-3 if args.model == "mobilenet" else 3e-3),
              "weight_decay": 1e-4, "label_smoothing": .05,
              "augmentation_config": {"horizontal_flip": .5, "vertical_flip": .5, "rotation_degrees": 25,
                                      "rotation_probability": .5, "crop_scale": [.75, 1], "color_jitter": [.25, .25, .25, .03]},
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (run_dir / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")

    t0 = time.time()
    model = build_model(args.model, len(classes), pretrained=True)
    model = model.to(memory_format=torch.channels_last)  # CPU-да конволюцияны жылдамдатады
    (run_dir / "model_summary.txt").write_text(f"{model}\n\nParameters: {count_params(model)}\nInput: 1x3x{size}x{size}\nOutput classes: {len(classes)}\n", encoding="utf-8")
    print("Параметрлер саны:", count_params(model))
    xte, yte = load_tensor(te, size)
    te_loader = DataLoader(TensorDataset(xte, yte), batch_size=128)

    if args.model != "mobilenet":
        # Нөлден оқыту: суреттер жадта, әр батчқа онлайн аугментация
        xtr, ytr = load_tensor(tr, size); xva, yva = load_tensor(va, size)
        tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.bs, shuffle=True)
        va_loader = DataLoader(TensorDataset(xva, yva), batch_size=128)
        net = model
        prep_train = lambda xb: normalize(augment(xb, size) if args.augmentation == "on" else xb)
        prep_eval = normalize
    else:
        # Трансферлік оқыту: MobileNetV2-нің алғашқы CUT блогы мұздатылады (ImageNet белгілері),
        # олардың шығысы бір рет есептеліп кэштеледі, ал жоғарғы блоктар мен классификатор оқытылады.
        stem = model.features[:args.cut].eval()
        for p in stem.parameters():
            p.requires_grad = False
        def feats(samples, aug):
            out = []
            for i in range(0, len(samples), 128):
                xb, _ = load_tensor(samples[i:i + 128], size)
                with torch.no_grad():
                    out.append(stem(normalize(augment(xb, size) if aug and args.augmentation == "on" else xb)).half())
            return torch.cat(out)
        ytr_ = torch.tensor([s[1] for s in tr]); yva = torch.tensor([s[1] for s in va])
        ftr = torch.cat([feats(tr, aug=v > 0) for v in range(args.views)])
        ytr = ytr_.repeat(args.views)
        fva = feats(va, aug=False)
        print("Кэштелген белгілер:", tuple(ftr.shape), f"{ftr.numel() * 2 / 1e6:.0f} МБ")
        net = Head(model, args.cut)
        tr_loader = DataLoader(TensorDataset(ftr, ytr), batch_size=args.bs, shuffle=True)
        va_loader = DataLoader(TensorDataset(fva, yva), batch_size=256)
        def prep_train(fb):
            fb = fb.float()
            if args.augmentation == "on" and random.random() < 0.5: fb = fb.flip(3)
            if args.augmentation == "on" and random.random() < 0.5: fb = fb.flip(2)
            return fb
        prep_eval = lambda fb: fb.float()
    prep_time = time.time() - t0
    print(f"Деректер дайындалды: {prep_time:.1f} с")

    lr = config["lr"]
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    params = [p for p in net.parameters() if p.requires_grad]
    print("Оқытылатын параметрлер:", sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=args.epochs, steps_per_epoch=len(tr_loader))

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
    best, best_epoch, best_val_acc, stale = -1.0, 0, 0.0, 0
    val_f1_history = []
    for ep in range(1, args.epochs + 1):
        net.train(); t = time.time()
        tot, correct, ls = 0, 0, 0.0
        for xb, yb in tr_loader:
            out = net(prep_train(xb)); loss = crit(out, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ls += loss.item() * len(yb); tot += len(yb); correct += (out.argmax(1) == yb).sum().item()
        vl, vacc, val_pred, val_gt, _ = evaluate(net, va_loader, prep_eval)
        val_f1 = float(f1_score(val_gt, val_pred, labels=list(range(len(classes))), average="macro", zero_division=0))
        val_f1_history.append(val_f1)
        dt = time.time() - t
        for k, v in zip(hist, [ls / tot, correct / tot, vl, vacc, dt]):
            hist[k].append(v)
        print(f"Дәуір {ep:2d}/{args.epochs} | train loss {ls/tot:.3f} acc {correct/tot:.3f} | val loss {vl:.3f} acc {vacc:.3f} | {dt:.0f} с", flush=True)
        if val_f1 > best:
            best, best_epoch, best_val_acc, stale = val_f1, ep, vacc, 0
            torch.save({"model": args.model, "state_dict": model.state_dict(), "classes": classes, "img_size": size,
                        "class_to_idx": {c:i for i,c in enumerate(classes)}, "normalization": config["normalization"],
                        "preprocessing": config["preprocessing"],
                        "run_id": run_id, "seed": args.seed, "best_epoch": best_epoch,
                        "selection_criterion": "validation_macro_f1", "dataset_manifest_sha256": manifest["manifest_sha256"],
                        "source_sha256": environment["source_sha256"]}, checkpoint_path)
        else:
            stale += 1
        with open(run_dir / "history.csv", "w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["epoch", *hist.keys(), "val_macro_f1"])
            writer.writerows([i+1, *[hist[k][i] for k in hist], val_f1_history[i]] for i in range(ep))
        if args.patience and stale >= args.patience:
            print(f"Early stopping at epoch {ep}; best validation Macro F1 epoch {best_epoch}", flush=True)
            break

    # Ең жақсы модельді тест жиынында бағалау
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["state_dict"])
    tl, tacc, pred, gt, prob = evaluate(model, te_loader)
    top3 = float(np.mean([g in np.argsort(-p)[:3] for g, p in zip(gt, prob)]))

    # Бір суретті болжау уақыты
    model.eval(); x1 = normalize(xte[:1])
    with torch.no_grad():
        for _ in range(5): model(x1)
        t = time.time()
        for _ in range(30): model(x1)
    infer_ms = (time.time() - t) / 30 * 1000

    res = {
        "model": args.model, "title": MODEL_SPECS[args.model]["title"], "img_size": size,
        "params": count_params(model), "trainable_params": sum(p.numel() for p in params),
        "frozen_blocks": args.cut if args.model == "mobilenet" else 0,
        "prep_time_s": prep_time, "train_samples": len(tr), "val_samples": len(va), "test_samples": len(te),
        "epochs": len(hist["train_loss"]), "planned_epochs": args.epochs, "best_epoch": best_epoch, "lr": lr, "batch_size": args.bs,
        "best_val_acc": best_val_acc, "best_val_macro_f1": best, "selection_criterion": "validation_macro_f1", "test_loss": tl, "test_acc": tacc, "test_top3": top3,
        "precision_macro": precision_score(gt, pred, labels=list(range(len(classes))), average="macro", zero_division=0),
        "recall_macro": recall_score(gt, pred, labels=list(range(len(classes))), average="macro", zero_division=0),
        "f1_macro": f1_score(gt, pred, labels=list(range(len(classes))), average="macro", zero_division=0),
        "infer_ms_cpu": infer_ms, "train_time_s": sum(hist["epoch_time"]),
        "model_size_mb": checkpoint_path.stat().st_size / 1e6,
        "history": hist,
        "val_macro_f1_history": val_f1_history, "run_id": run_id, "seed": args.seed,
        "purpose": config["purpose"],
        "checkpoint_sha256": sha256_file(checkpoint_path), "dataset_manifest_sha256": manifest["manifest_sha256"],
        "environment": environment, "calibrated": False, "external_evaluation": "not_performed",
        "report": classification_report(gt, pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(gt, pred, labels=list(range(len(classes)))).tolist(),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"Artifacts: {run_dir}; deployment checkpoints were not replaced")
    print(f"ТЕСТ: acc {tacc:.4f} | top3 {top3:.4f} | F1 {res['f1_macro']:.4f} | {infer_ms:.1f} мс/сурет")


if __name__ == "__main__":
    main()
