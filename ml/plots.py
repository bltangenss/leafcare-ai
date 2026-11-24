"""Курстық жұмысқа арналған суреттер мен графиктерді құру (reports/ папкасына)."""
import argparse, json, os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
DATA = os.path.join(ROOT, "data", "plantvillage")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
C1, C2 = "#2f8f46", "#d0743c"


def load(name):
    current = os.path.join(REP, "evaluation_current", f"{name}_evaluation.json")
    if os.path.exists(current):
        from ml.dataset import sha256_file
        with open(current, encoding="utf8") as source:
            result = json.load(source)
        if result.get("checkpoint_sha256") == sha256_file(os.path.join(ROOT, "models", f"{name}.pt")):
            return result
    p = os.path.join(REP, f"{name}_results.json")
    return json.load(open(p, encoding="utf8")) if os.path.exists(p) else None


def short(cls):
    from app.disease_info import get_info
    i = get_info(cls)
    label = f"{i['plant']}: {i['name']}"
    return label if len(label) <= 42 else label[:41].rstrip() + "…"


def curves(r):
    if not r.get("history"):
        return  # Checkpoints do not contain an invented training history.
    h = r["history"]; ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, h["train_loss"], "-o", ms=3, c=C1, label="Оқыту"); ax[0].plot(ep, h["val_loss"], "-o", ms=3, c=C2, label="Валидация")
    ax[0].set_title("Шығын функциясы (Loss)"); ax[0].set_xlabel("Дәуір"); ax[0].legend()
    ax[1].plot(ep, np.array(h["train_acc"]) * 100, "-o", ms=3, c=C1, label="Оқыту")
    ax[1].plot(ep, np.array(h["val_acc"]) * 100, "-o", ms=3, c=C2, label="Валидация")
    ax[1].set_title("Дәлдік (Accuracy), %"); ax[1].set_xlabel("Дәуір"); ax[1].legend()
    for a in ax: a.grid(alpha=.3)
    fig.suptitle(r["title"], fontweight="bold")
    fig.savefig(os.path.join(REP, f"{r['model']}_curves.png")); plt.close(fig)


def confusion(r, classes):
    cm = np.array(r["confusion"], dtype=float)
    cmn = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) != 0)
    fig, ax = plt.subplots(figsize=(13, 11.5))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    labels = [short(c) for c in classes]
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(range(1, len(classes) + 1), fontsize=7)
    ax.set_yticklabels([f"{i+1}. {l}" for i, l in enumerate(labels)], fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=5.5, color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("Болжанған класс (нөмірі)"); ax.set_ylabel("Нақты класс")
    ax.set_title(f"Қателіктер матрицасы — {r['title']}", fontweight="bold")
    fig.colorbar(im, fraction=.03)
    fig.savefig(os.path.join(REP, f"{r['model']}_confusion.png")); plt.close(fig)


def per_class_f1(rs, classes):
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(classes)); hgt = 0.4
    for k, (r, c) in enumerate(zip(rs, [C2, C1])):
        f1 = [r["report"][cl]["f1-score"] * 100 for cl in classes]
        ax.barh(y + (k - .5) * hgt, f1, height=hgt, color=c, label=r["title"])
    ax.set_yticks(y); ax.set_yticklabels([short(c) for c in classes], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlim(0, 101); ax.set_xlabel("F1-өлшем, %"); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.055), ncol=2, fontsize=9)
    ax.set_title("Әр класс бойынша F1-өлшем", fontweight="bold"); ax.grid(axis="x", alpha=.3)
    fig.savefig(os.path.join(REP, "per_class_f1.png")); plt.close(fig)


def compare(rs):
    keys = [("test_acc", "Accuracy"), ("precision_macro", "Precision"), ("recall_macro", "Recall"), ("f1_macro", "F1"), ("test_top3", "Top-3")]
    fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(keys)); w = .36
    for k, (r, c) in enumerate(zip(rs, [C2, C1])):
        v = [r[a] * 100 for a, _ in keys]
        b = ax.bar(x + (k - .5) * w, v, w, color=c, label=r["title"])
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys]); ax.set_ylim(0, 106)
    ax.set_ylabel("%"); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, fontsize=9); ax.set_title("Модельдерді салыстыру (тест жиыны)", fontweight="bold")
    fig.savefig(os.path.join(REP, "compare.png")); plt.close(fig)


def samples(classes):
    random.seed(3)
    pick = random.sample(classes, 12)
    fig, axs = plt.subplots(3, 4, figsize=(10, 8))
    for a, c in zip(axs.flat, pick):
        f = random.choice(os.listdir(os.path.join(DATA, c)))
        a.imshow(Image.open(os.path.join(DATA, c, f))); a.axis("off")
        a.set_title(short(c).replace(": ", ":\n"), fontsize=8.5)
    fig.suptitle("PlantVillage датасетінен мысалдар", fontweight="bold")
    fig.savefig(os.path.join(REP, "samples.png")); plt.close(fig)


def distribution():
    p = os.path.join(REP, "full_counts.json")
    if not os.path.exists(p): return
    cnt = json.load(open(p, encoding="utf8"))
    items = sorted(cnt.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh([short(k) for k, _ in items], [v for _, v in items], color=C1)
    ax.axvline(250, ls="--", c=C2, label="Ішкі жиын: 250 сурет/класс")
    ax.set_xlabel("Суреттер саны"); ax.tick_params(axis="y", labelsize=7.5); ax.legend()
    ax.set_title(f"PlantVillage толық датасетіндегі кластар үлестірімі (барлығы {sum(cnt.values())})", fontweight="bold")
    fig.savefig(os.path.join(REP, "distribution.png")); plt.close(fig)


def box(ax, x, y, w, h, text, fc, fs=8.5, ec="#333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arch_plantcnn():
    from ml.models import WIDTHS, HIDDEN
    fig, ax = plt.subplots(figsize=(12, 3.2)); ax.set_xlim(0, 12.2); ax.set_ylim(0, 3); ax.axis("off")
    greens = ["#d8efdc", "#c4e6ca", "#aedcb6", "#93cf9e"]
    blocks = [("Кіріс\n128×128×3", "#e8eef7")]
    sz = 128
    for i, w in enumerate(WIDTHS):
        sz //= 2
        blocks.append((f"Блок {i+1}\n2×Conv3×3({w})\nBN+ReLU\nMaxPool\n→{sz}×{sz}×{w}", greens[i]))
    blocks += [(f"Global\nAvgPool\n→{WIDTHS[-1]}", "#fde7c8"), (f"Dropout 0.4\nFC {HIDDEN}\nReLU\nDropout 0.3", "#fbd4ae"),
               ("FC 38\nSoftmax", "#f6b98a")]
    x = 0.1
    for i, (t, c) in enumerate(blocks):
        box(ax, x, 0.5, 1.3, 2.0, t, c)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + 1.5, 1.5), xytext=(x + 1.3, 1.5), arrowprops=dict(arrowstyle="->", lw=1.3))
        x += 1.5
    ax.set_title("PlantCNN архитектурасы", fontweight="bold")
    fig.savefig(os.path.join(REP, "arch_plantcnn.png")); plt.close(fig)


def arch_transfer():
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.set_xlim(0, 12.2); ax.set_ylim(0, 3.6); ax.axis("off")
    blocks = [("Кіріс\n192×192×3", "#e8eef7", 1.3), ("Conv 3×3\nstride 2\n96×96×32", "#dfe7f1", 1.3),
              ("Inverted\nresidual\nблоктар 1–13\n→12×12×96", "#dfe7f1", 1.7),
              ("Inverted\nresidual\nблоктар 14–17\n→6×6×320", "#93cf9e", 1.7), ("Conv 1×1\n→6×6×1280", "#93cf9e", 1.3),
              ("GAP\n→1280", "#fde7c8", 1.1), ("Dropout 0.3\nFC 38\nSoftmax", "#f6b98a", 1.4)]
    x = 0.1
    for i, (t, c, w) in enumerate(blocks):
        box(ax, x, 0.7, w, 2.0, t, c)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + w + 0.2, 1.7), xytext=(x + w, 1.7), arrowprops=dict(arrowstyle="->", lw=1.3))
        x += w + 0.2
    ax.add_patch(FancyBboxPatch((0.05, 0.25), 4.75, 2.75, boxstyle="round,pad=0.02", fc="none", ec="#4a6fa5", lw=1.5, ls="--"))
    ax.text(2.45, 0.38, "Мұздатылған (ImageNet салмақтары), шығысы кэштеледі", ha="center", fontsize=8.5, color="#4a6fa5")
    ax.add_patch(FancyBboxPatch((4.95, 0.25), 7.15, 2.75, boxstyle="round,pad=0.02", fc="none", ec=C1, lw=1.5, ls="--"))
    ax.text(8.5, 0.38, "Оқытылатын бөлік (fine-tuning)", ha="center", fontsize=8.5, color=C1)
    ax.set_title("MobileNetV2 негізіндегі трансферлік оқыту", fontweight="bold")
    fig.savefig(os.path.join(REP, "arch_transfer.png")); plt.close(fig)


def arch_system():
    fig, ax = plt.subplots(figsize=(13, 5.1))
    ax.set_xlim(-.1, 12.4); ax.set_ylim(0, 5); ax.axis("off")
    main = [
        (0, "React / TypeScript\nСурет және preview", "#e8eef7"),
        (2.5, "FastAPI REST\nФайлды тексеру", "#d8efdc"),
        (5, "RGB / Resize\nНормализация", "#fde7c8"),
        (7.5, "PyTorch CNN\nЫқтимал кластар", "#aedcb6"),
        (10, "JSON нәтиже\nНұсқа және ескерту", "#e8eef7"),
    ]
    arrow = dict(arrowstyle="->", lw=1.3)
    for x, text, color in main:
        box(ax, x, 3.2, 2.1, 1.1, text, color, 9)
        if x < 10:
            ax.annotate("", xy=(x+2.5, 3.75), xytext=(x+2.1, 3.75), arrowprops=arrow)
    box(ax, 0, .5, 2.1, 1.1, "SQLite тарихы\nТек келісіммен", "#e3eef7", 9)
    box(ax, 5, .5, 2.1, 1.1, "Checkpoint\nClass mapping\nPreprocessing", "#eeeeee", 9)
    box(ax, 7.5, .5, 2.1, 1.1, "Grad-CAM\nТүсіндірме картасы", "#fbd4ae", 9)
    box(ax, 10, .5, 2.1, 1.1, "Анықтамалық\nҚазақша атаулар", "#f3e3f7", 9)
    ax.annotate("", xy=(8.55, 3.2), xytext=(6.05, 1.6), arrowprops=arrow)
    ax.annotate("", xy=(8.55, 1.6), xytext=(8.55, 3.2), arrowprops=arrow)
    ax.annotate("", xy=(11.05, 3.2), xytext=(9.6, 1.6), arrowprops=arrow)
    ax.annotate("", xy=(11.05, 3.2), xytext=(11.05, 1.6), arrowprops=arrow)
    ax.annotate("", xy=(1.05, 1.6), xytext=(3.55, 3.2), arrowprops=arrow)
    ax.set_title("LeafCare AI веб-қосымшасының архитектурасы", fontweight="bold")
    fig.savefig(os.path.join(REP, "arch_system.png")); plt.close(fig)


def gradcam_examples(classes, model="mobilenet"):
    import torch
    from app.predictor import Predictor
    pr = Predictor(); random.seed(7)
    pick = [c for c in classes if "healthy" not in c]
    pick = random.sample(pick, 5)
    fig, axs = plt.subplots(2, 5, figsize=(12, 5.2))
    import base64, io
    for k, c in enumerate(pick):
        f = random.choice(os.listdir(os.path.join(DATA, c)))
        img = Image.open(os.path.join(DATA, c, f)).convert("RGB")
        r = pr.predict(img, model)
        cam = Image.open(io.BytesIO(base64.b64decode(r["heatmap"].split(",")[1])))
        axs[0, k].imshow(img); axs[0, k].set_title(short(c).replace(": ", ":\n"), fontsize=8); axs[0, k].axis("off")
        axs[1, k].imshow(cam); axs[1, k].axis("off")
        axs[1, k].set_title(f"Болжам: {r['top'][0]['prob']*100:.1f}%\n" + short(r["top"][0]["cls"]).split(": ")[1][:30], fontsize=8)
    fig.suptitle("Grad-CAM: модель шешім қабылдауда назар аударған аймақтар", fontweight="bold")
    fig.savefig(os.path.join(REP, "gradcam_examples.png")); plt.close(fig)


def main():
    if hasattr(os.sys.stdout, "reconfigure"):
        os.sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradcam", action="store_true", help="Also generate five Grad-CAM examples")
    args = parser.parse_args()
    os.makedirs(REP, exist_ok=True)
    split = json.load(open(os.path.join(REP, "split.json"), encoding="utf8")); classes = split["classes"]
    rs = [r for r in (load("plantcnn"), load("mobilenet")) if r]
    for r in rs:
        curves(r); confusion(r, classes)
    # Only the actual saved run has curves. An incomplete PlantCNN run does not.
    for name in ("plantcnn", "mobilenet"):
        path = os.path.join(REP, f"{name}_results.json")
        if os.path.exists(path):
            with open(path, encoding="utf8") as source:
                curves(json.load(source))
    if len(rs) == 2:
        per_class_f1(rs, classes); compare(rs)
    samples(classes); distribution(); arch_plantcnn(); arch_transfer(); arch_system()
    if rs and args.gradcam:
        gradcam_examples(classes, rs[-1]["model"])
    print("Суреттер дайын:", sorted(f for f in os.listdir(REP) if f.endswith(".png")))


if __name__ == "__main__":
    main()
