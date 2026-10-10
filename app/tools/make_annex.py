"""Génère les éléments de l'annexe « évaluation en conditions réelles » du mémoire.

Seules les contributions dont le consentement « mémoire » est accordé sont utilisées.
Vérité de référence : l'émotion demandée (défi) ou l'émotion indiquée par la personne.

Usage :
    python app/tools/make_annex.py --data <dossier des contributions ou export dézippé> --out annexe/
Produit :
    planche.png           grille d'exemples (visage, émotion de référence, prédictions R/V/D)
    confusion_<m>.png     matrice de confusion de chaque modèle
    resultats.csv         une ligne par image
    resume.md             exactitude par modèle, nombre de participants et d'images
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FR = {"angry": "Colère", "disgust": "Dégoût", "fear": "Peur", "happy": "Joie", "neutral": "Neutre", "sad": "Tristesse", "surprise": "Surprise"}
MODELS = {"resnet": "ResNet-50", "vit": "ViT-Base", "dbfn": "DBFN"}


def load(data_dir):
    root = Path(data_dir)
    base = root / "contributions" if (root / "contributions").is_dir() else root
    rows = []
    for meta_path in sorted(base.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("consent", {}).get("memoire"):
            continue
        for it in meta["items"]:
            ref = it.get("target") or it.get("true_label")
            preds = it.get("predictions") or {}
            if ref not in CLASSES or not all(m in preds for m in MODELS):
                continue
            rows.append({"contribution": meta["id"], "image": meta_path.parent / it["file"], "source": it.get("source"), "ref": ref,
                         **{m: preds[m]["top"] for m in MODELS}, **{f"{m}_conf": preds[m]["confidence"] for m in MODELS}})
    return rows


def confusion(rows, m, out):
    cm = np.zeros((7, 7))
    for r in rows:
        cm[CLASSES.index(r["ref"]), CLASSES.index(r[m])] += 1
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [FR[c] for c in CLASSES]
    ax.set_xticks(range(7), labels, rotation=45, ha="right"); ax.set_yticks(range(7), labels)
    for i in range(7):
        for j in range(7):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if norm[i, j] > .5 else "black", fontsize=9)
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Référence")
    acc = np.trace(cm) / max(cm.sum(), 1) * 100
    ax.set_title(f"{MODELS[m]} — exactitude {acc:.1f} % (n = {int(cm.sum())})")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return acc


def sheet(rows, out, per_row=7, cell=220, max_items=42):
    rows = rows[:max_items]
    if not rows:
        return
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    n_rows = (len(rows) + per_row - 1) // per_row
    W, H = per_row * cell, n_rows * (cell + 58)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for k, r in enumerate(rows):
        x, y = (k % per_row) * cell, (k // per_row) * (cell + 58)
        img.paste(Image.open(r["image"]).convert("RGB").resize((cell - 8, cell - 8)), (x + 4, y + 4))
        d.text((x + 6, y + cell - 2), f"Réf. : {FR[r['ref']]}", fill="black", font=font)
        marks = " ".join(f"{m[0].upper()}{'✓' if r[m] == r['ref'] else '✗'}" for m in MODELS)
        d.text((x + 6, y + cell + 22), marks, fill=(30, 30, 30), font=font)
    img.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="annexe")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = load(args.data)
    if not rows:
        print("Aucune contribution exploitable (consentement mémoire + étiquette de référence)."); return
    with (out / "resultats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "image"] + ["image"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "image": str(r["image"])})
    accs = {m: confusion(rows, m, out / f"confusion_{m}.png") for m in MODELS}
    sheet(rows, out / "planche.png")
    people = len({r["contribution"] for r in rows})
    lines = ["# Évaluation en conditions réelles", "", f"- Contributions (participants) : {people}", f"- Images : {len(rows)}", "",
             "| Modèle | Exactitude |", "|---|---:|"] + [f"| {MODELS[m]} | {a:.1f} % |" for m, a in accs.items()]
    (out / "resume.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
