#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_affectnet7.py — Construit data_affectnet_fer/ (AffectNet-7, format FER2013)
=================================================================================
- Lit les labels depuis les .pt officiels (annotations "<stem>_exp", 0-7).
- Retire la classe 7 (contempt) -> AffectNet-7 (7 classes, cohérent FER2013/RAF-DB).
- Crée des liens symboliques images -> data_affectnet_fer/<split>/<emotion>/<file>.
  train_dataset.pt -> split "train" ; test_dataset.pt (val_set) -> split "test".
"""
import os, torch, __main__
from collections import Counter

class AffectNetClassificationDataset:  # stub pour dépickler
    pass
__main__.AffectNetClassificationDataset = AffectNetClassificationDataset

# AffectNet exp -> nom d'émotion FER (7 recognized; 7=contempt supprimé)
EXP2NAME = {0: "neutral", 1: "happy", 2: "sad", 3: "surprise",
            4: "fear", 5: "disgust", 6: "angry"}  # 7: contempt -> drop

BASE = os.path.abspath("data_affectnet/AffectNet/AffectNet-images")
PT = "/tmp/affnet_pt/AffectNet/AffectNet-torch"
OUT = os.path.abspath("data_affectnet_fer")

SPLITS = [
    ("train", f"{PT}/train_dataset.pt", f"{BASE}/train_set/images"),
    ("test",  f"{PT}/test_dataset.pt",  f"{BASE}/val_set/images"),
]

for split, pt_path, img_dir in SPLITS:
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    files = obj.__dict__["image_files"]
    ann = obj.__dict__["annotations"]
    kept, dropped, missing = 0, 0, 0
    cnt = Counter()
    for f in files:
        stem = f.split(".")[0]
        exp = ann.get(f"{stem}_exp")
        if exp is None:
            missing += 1; continue
        exp = int(exp)
        if exp == 7:  # contempt
            dropped += 1; continue
        name = EXP2NAME[exp]
        src = os.path.join(img_dir, f)
        dst_dir = os.path.join(OUT, split, name)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
        kept += 1; cnt[name] += 1
    print(f"[{split}] gardés={kept} | contempt retirés={dropped} | annot. manquantes={missing}")
    print(f"  par classe: {dict(sorted(cnt.items()))}")

print("OK -> data_affectnet_fer/")
