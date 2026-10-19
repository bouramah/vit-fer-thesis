"""Taux d'avertissements de qualité par condition (sans modèle) : détection et fausses alertes.

Usage, depuis la racine du dépôt :
    PYTHONPATH=. python app/tools/quality_calibration.py --per_class 100
"""
import argparse
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.server.quality import face_quality
from app.tools.robustness_test import CONDITIONS, degrade, landmarks

EXPECTED = {"clear_glasses": None, "sunglasses": "eyes_hidden", "mask": "mouth_hidden", "blur": "blur",
            "low_light": "dark", "noise": "noise", "small": "small"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fer2013/test")
    ap.add_argument("--yunet", default="app/weights/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--per_class", type=int, default=100)
    args = ap.parse_args()
    random.seed(7)
    rng = np.random.default_rng(7)
    det = cv2.FaceDetectorYN.create(args.yunet, "", (240, 240), 0.5)
    data = []
    for c in sorted(os.listdir(args.data)):
        if c.startswith("."):
            continue
        files = sorted(os.listdir(Path(args.data) / c))
        for f in random.sample(files, min(args.per_class, len(files))):
            g = np.array(Image.open(Path(args.data) / c / f).convert("L"))
            lm = landmarks(det, g)
            if lm is not None:
                data.append((g, lm))
    print(f"{len(data)} images")
    for cond in CONDITIONS:
        cnt, anyw = Counter(), 0
        for g, lm in data:
            x = degrade(g, cond, lm, rng)
            w = face_quality(Image.fromarray(x).convert("RGB"), lm, face_px=16 if cond == "small" else 48)["warnings"]
            cnt.update(w)
            anyw += bool(w)
        n = len(data)
        exp = EXPECTED.get(cond)
        hit = f"  détecté ({exp}) {100 * cnt[exp] / n:5.1f} %" if exp else ""
        print(f"{cond:14s} au moins un avertissement {100 * anyw / n:5.1f} %{hit}  |  " +
              ", ".join(f"{k} {100 * v / n:.1f}" for k, v in cnt.most_common()))


if __name__ == "__main__":
    main()
