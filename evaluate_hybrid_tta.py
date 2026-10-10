#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_hybrid_tta.py — Évaluation DBFN (Hybrid) avec TTA sur FER2013
======================================================================
- Charge un checkpoint DBFN (concat ou cross_attention)
- Évalue sur data/fer2013/test
- Applique TTA (none | hflip | 5crop | 10crop)
- Moyenne les logits (plus stable que moyenne des probas)

Requis:
- models.hybrid.create_dbfn
- utils.dataset.FER2013Dataset, FER2013_CLASSES
- utils.metrics.print_classification_report (+ plots si tu veux)

Auteur: adapté pour Boss
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torchvision import transforms
from PIL import Image

from models.hybrid import create_dbfn
from utils.dataset import FER2013Dataset, FER2013_CLASSES
from utils.metrics import (
    print_classification_report,
    plot_confusion_matrix,
    plot_per_class_accuracy,
)

# ----------------------------
# Utils
# ----------------------------

def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[INFO] MPS (Apple GPU)")
            return torch.device("mps")
        else:
            print("[INFO] CPU")
            return torch.device("cpu")
    return torch.device(device_str)


def load_checkpoint_state(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # compat: parfois ckpt est directement state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = {k: ckpt.get(k, None) for k in ["epoch", "val_f1"]}
    else:
        state = ckpt
        meta = {"epoch": None, "val_f1": None}
    return state, meta


# ----------------------------
# TTA transforms
# ----------------------------

def build_base_preprocess(img_size: int = 224):
    """
    Préprocess identique à ton val_transforms:
    Resize -> Grayscale(3) -> ToTensor -> Normalize(ImageNet)
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


def apply_tta_pil(img_pil: Image.Image, tta: str, img_size: int = 224):
    """
    Retourne une liste de tenseurs [C,H,W] (déjà normalisés).
    """
    base = build_base_preprocess(img_size)

    if tta == "none":
        return [base(img_pil)]

    if tta == "hflip":
        imgs = [
            img_pil,
            transforms.functional.hflip(img_pil),
        ]
        return [base(im) for im in imgs]

    # Pour crop TTA, on fait crop AVANT resize final
    # (sinon crop sur image déjà 224 ne sert presque à rien)
    # Strategy: Resize plus grand -> FiveCrop -> chaque crop -> Resize -> preprocess final
    up = int(img_size * 1.12)  # ex 224 -> 250
    resize_up = transforms.Resize((up, up))
    to_gray = transforms.Grayscale(num_output_channels=3)

    def preprocess_crop(crop_pil):
        crop_pil = transforms.Resize((img_size, img_size))(crop_pil)
        crop_pil = to_gray(crop_pil)
        crop_t = transforms.ToTensor()(crop_pil)
        crop_t = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )(crop_t)
        return crop_t

    if tta == "5crop":
        img_up = resize_up(img_pil)
        crops = transforms.FiveCrop(size=img_size)(img_up)  # tuple de 5 PIL
        return [preprocess_crop(c) for c in crops]

    if tta == "10crop":
        img_up = resize_up(img_pil)
        crops = transforms.TenCrop(size=img_size)(img_up)  # 10 PIL (5 + 5 flips)
        return [preprocess_crop(c) for c in crops]

    raise ValueError("tta doit être: none | hflip | 5crop | 10crop")


@torch.no_grad()
def predict_batch_tta(model, images_batch, device, tta: str, img_size: int):
    """
    images_batch: liste de PIL Images (taille B)
    Retourne logits moyens: Tensor [B, num_classes]
    """
    # On construit toutes les versions TTA de toutes les images du batch
    # puis on fait N forward et on moyenne les logits.
    # Comme chaque image peut produire K crops, on gère par boucle sur images.
    logits_accum = []
    num_classes = 7

    for img_pil in images_batch:
        tta_tensors = apply_tta_pil(img_pil, tta=tta, img_size=img_size)  # list of [C,H,W]
        x = torch.stack(tta_tensors, dim=0).to(device)  # [K,C,H,W]
        out = model(x)  # [K,7]
        out_mean = out.mean(dim=0)  # [7]
        logits_accum.append(out_mean)

    return torch.stack(logits_accum, dim=0)  # [B,7]


# ----------------------------
# Main eval
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/fer2013")
    ap.add_argument("--checkpoint", type=str, required=True)

    ap.add_argument("--fusion", type=str, default="concat", choices=["concat", "cross_attention"])
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--drop_path_rate", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])

    ap.add_argument("--tta", type=str, default="hflip", choices=["none", "hflip", "5crop", "10crop"])
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--out_dir", type=str, default="results_tta")

    args = ap.parse_args()

    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Dataset test (IMPORTANT: transform=None car on fait nous-mêmes le preprocess + TTA)
    test_ds = FER2013Dataset(root_dir=args.data_dir, split="test", transform=None, return_path=False)

    # 2) DataLoader: on charge des PIL + labels
    def collate_pil(batch):
        # batch = [(image_pil, label), ...]
        imgs = [x[0] for x in batch]
        labs = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return imgs, labs

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pil
    )

    # 3) Modèle DBFN (on le reconstruit comme pendant l'entraînement)
    print(f"\n[MODEL] create_dbfn fusion={args.fusion} proj_dim={args.proj_dim}")
    model = create_dbfn(
        cnn_checkpoint=None,   # déjà inclus dans le checkpoint final DBFN
        vit_checkpoint=None,   # idem
        fusion_type=args.fusion,
        freeze_backbones=False,   # en eval ça ne change rien
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        num_classes=7,
    ).to(device)
    model.eval()

    # 4) Charger checkpoint
    state, meta = load_checkpoint_state(args.checkpoint, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] Loaded: {args.checkpoint}")
    print(f"  epoch={meta.get('epoch')} val_f1={meta.get('val_f1')}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"  missing ex: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"  unexpected ex: {unexpected[:5]}")

    # 5) Inference TTA
    all_preds = []
    all_labels = []

    pbar = tqdm(test_loader, desc=f"Eval TTA={args.tta}")
    for imgs_pil, labels in pbar:
        labels = labels.to(device, non_blocking=True)

        logits = predict_batch_tta(model, imgs_pil, device, tta=args.tta, img_size=args.img_size)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = 100.0 * np.mean(all_preds == all_labels)

    from sklearn.metrics import f1_score
    f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n" + "="*70)
    print(f"✅ DBFN + TTA={args.tta} — TEST")
    print("="*70)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Macro-F1:  {f1:.2f}%")

    print_classification_report(all_labels, all_preds, f"DBFN_{args.fusion}_TTA_{args.tta}")

    # Figures
    cm_path = os.path.join(args.out_dir, f"cm_dbfn_{args.fusion}_tta_{args.tta}.png")
    acc_path = os.path.join(args.out_dir, f"acc_dbfn_{args.fusion}_tta_{args.tta}.png")

    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path,
                          title=f"Confusion — DBFN {args.fusion} TTA={args.tta} (Test)")
    plot_per_class_accuracy(all_labels, all_preds, save_path=acc_path,
                            title=f"Per-class Acc — DBFN {args.fusion} TTA={args.tta} (Test)")

    print(f"\n[OUT] Saved in: {args.out_dir}")


if __name__ == "__main__":
    main()