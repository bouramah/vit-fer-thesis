"""Robustesse d'EmoVision aux occultations (lunettes, masque) et à la qualité d'image.

Applique des dégradations synthétiques aux images de test de FER2013, puis mesure
l'exactitude de ResNet-50, ViT-Base et du DBFN (les poids d'EmoVision) pour chacune.
Les occultations sont placées à partir des points du visage détectés par YuNet.

Usage, depuis la racine du dépôt :
    PYTHONPATH=. python app/tools/robustness_test.py --data data/fer2013/test --out results_robustness
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.server.engine import Engine
from app.server.quality import face_quality

CONDITIONS = ["clean", "clear_glasses", "sunglasses", "mask", "blur", "low_light", "noise", "small"]


def landmarks(det, gray):
    """Points YuNet (œil droit, œil gauche, nez, coin droit, coin gauche de la bouche) sur l'image d'origine."""
    up = cv2.resize(gray, (240, 240), interpolation=cv2.INTER_CUBIC)
    bgr = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    det.setInputSize((240, 240))
    _, faces = det.detect(bgr)
    if faces is None:
        return None
    f = max(faces, key=lambda r: r[14])
    return (f[4:14].reshape(5, 2) * gray.shape[0] / 240.0).astype(np.float32)


def degrade(gray, cond, lm, rng):
    img = gray.astype(np.float32)
    s = gray.shape[0]
    reye, leye, nose, rmouth, lmouth = lm
    d = float(np.linalg.norm(leye - reye))
    if cond == "clean":
        return gray
    if cond in ("clear_glasses", "sunglasses"):
        out = gray.copy()
        axes = (int(0.42 * d), int(0.30 * d))
        for c in (reye, leye):
            center = (int(c[0]), int(c[1]))
            if cond == "sunglasses":
                cv2.ellipse(out, center, axes, 0, 0, 360, int(rng.uniform(10, 35)), -1)
            cv2.ellipse(out, center, axes, 0, 0, 360, 25, max(1, int(0.06 * d)))
        bridge = (int((reye[0] + leye[0]) / 2), int((reye[1] + leye[1]) / 2))
        cv2.line(out, (int(reye[0] + axes[0]), bridge[1]), (int(leye[0] - axes[0]), bridge[1]), 25, max(1, int(0.06 * d)))
        return out
    if cond == "mask":
        out = gray.copy()
        eyes_y = (reye[1] + leye[1]) / 2
        mouth_y = (rmouth[1] + lmouth[1]) / 2
        top = eyes_y + 0.55 * (nose[1] - eyes_y)
        bottom = min(s - 1, mouth_y + 1.1 * (mouth_y - nose[1]) + 0.25 * d)
        left, right = reye[0] - 0.75 * d, leye[0] + 0.75 * d
        pts = np.array([[left, top + 0.1 * d], [nose[0], top], [right, top + 0.1 * d],
                        [right - 0.15 * d, bottom - 0.2 * d], [nose[0], bottom], [left + 0.15 * d, bottom - 0.2 * d]], np.int32)
        cv2.fillPoly(out, [pts], int(rng.uniform(170, 225)))
        for k in (0.35, 0.55, 0.75):  # plis du masque chirurgical
            y = int(top + k * (bottom - top))
            cv2.line(out, (int(left + 0.1 * d), y), (int(right - 0.1 * d), y), int(rng.uniform(140, 165)), 1)
        return out
    if cond == "blur":
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=0.025 * s)
    if cond == "low_light":
        dark = 255.0 * (img / 255.0) ** 1.8 * 0.35
        return np.clip(dark + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    if cond == "noise":
        return np.clip(img + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8)
    if cond == "small":
        tiny = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        return cv2.resize(tiny, (s, s), interpolation=cv2.INTER_LINEAR)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fer2013/test")
    ap.add_argument("--weights", default="app/weights/emovision_weights.pt")
    ap.add_argument("--yunet", default="app/weights/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--out", default="results_robustness")
    ap.add_argument("--per_class", type=int, default=0, help="0 = tout le jeu de test")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    random.seed(42)

    eng = Engine(args.weights, threads=os.cpu_count())
    det = cv2.FaceDetectorYN.create(args.yunet, "", (240, 240), 0.5)
    items = []
    for ci, cls in enumerate(eng.classes):
        files = sorted(os.listdir(Path(args.data) / cls))
        if args.per_class:
            files = random.sample(files, min(args.per_class, len(files)))
        items += [(Path(args.data) / cls / f, ci) for f in files]

    # Points du visage : YuNet, sinon la moyenne des images où il a réussi.
    grays, lms = [], []
    for p, _ in items:
        g = np.array(Image.open(p).convert("L"))
        grays.append(g)
        lms.append(landmarks(det, g))
    found = [l / g.shape[0] for l, g in zip(lms, grays) if l is not None]
    mean_lm = np.mean(found, axis=0)
    n_found = len(found)
    lms = [l if l is not None else mean_lm * g.shape[0] for l, g in zip(lms, grays)]
    print(f"{len(items)} images, points YuNet trouvés pour {n_found}")

    labels = np.array([c for _, c in items])
    summary, rows = {}, []
    for cond in CONDITIONS:
        t0 = time.time()
        imgs = [degrade(g, cond, l, rng) for g, l in zip(grays, lms)]
        preds = {m: [] for m in ("resnet", "vit", "dbfn")}
        conf = []
        for i in range(0, len(imgs), args.batch):
            chunk = [Image.fromarray(x).convert("RGB") for x in imgs[i:i + args.batch]]
            probs, _ = eng.predict(chunk, "compare")
            for m in preds:
                preds[m].append(probs[m])
        res = {}
        for m in preds:
            p = np.concatenate(preds[m])
            pred = p.argmax(1)
            acc = float((pred == labels).mean())
            f1s = []
            for c in range(len(eng.classes)):
                tp = ((pred == c) & (labels == c)).sum()
                prec = tp / max(1, (pred == c).sum())
                rec = tp / max(1, (labels == c).sum())
                f1s.append(0 if tp == 0 else 2 * prec * rec / (prec + rec))
            res[m] = {"accuracy": round(100 * acc, 2), "macro_f1": round(100 * float(np.mean(f1s)), 2),
                      "mean_confidence": round(100 * float(p.max(1).mean()), 2)}
            if m == "dbfn":
                conf = p.max(1)
        summary[cond] = res
        for k, (x, l) in enumerate(zip(imgs, lms)):
            q = face_quality(Image.fromarray(x).convert("RGB"), l, face_px=x.shape[0] if cond != "small" else 16)
            rows.append({"cond": cond, "idx": k, "label": int(labels[k]), "dbfn_conf": round(float(conf[k]), 4), **q})
        ex = [Image.fromarray(imgs[k]).resize((120, 120)) for k in range(0, len(imgs), max(1, len(imgs) // 8))[:8]]
        sheet = Image.new("L", (120 * len(ex), 120))
        for k, e in enumerate(ex):
            sheet.paste(e, (120 * k, 0))
        sheet.save(out / "examples" / f"{cond}.png")
        print(f"{cond:14s} " + "  ".join(f"{m} {r['accuracy']:.2f}" for m, r in res.items()) + f"  ({time.time() - t0:.0f} s)")

    json.dump({"n_images": len(items), "landmarks_found": n_found, "results": summary},
              open(out / "summary.json", "w"), indent=2)
    with open(out / "quality_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
