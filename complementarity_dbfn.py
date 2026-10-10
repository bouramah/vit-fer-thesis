#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
complementarity_dbfn.py — Preuve QUANTITATIVE de complémentarité (remplace le t-SNE)
====================================================================================
Évalue ResNet-50 seul, ViT-Base seul et DBFN-concat sur le test FER2013, puis :
  - Décompose l'accord entre branches : both✓ / only-ResNet✓ / only-ViT✓ / both✗
  - Sur le sous-ensemble "litigieux" (exactement une branche a raison), mesure si
    le DBFN tranche correctement → preuve directe que la fusion exploite les deux.
  - Figure finale = barres de complémentarité (haut) + Grad-CAM ResNet (bas).

Usage:
  CUDA_VISIBLE_DEVICES=2 venv_fer/bin/python complementarity_dbfn.py
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.resnet import create_resnet50
from models.vit import ViTForFER
from models.hybrid import create_dbfn
from utils.dataset import FER2013Dataset, FER2013_CLASSES
from analyze_dbfn import GradCAM, denorm  # réutilise le Grad-CAM validé

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESNET_CKPT = "checkpoints_run3_resnet50/resnet50_v2_epoch61_best.pth"
VIT_CKPT = "checkpoints_vit_base_v3_run1/vit_base_v3_epoch86_best.pth"
DBFN_CKPT = "checkpoints_hybrid/dbfn_concat_frozen_20260227_173625_epoch1_best.pth"


def preprocess(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_state(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] {os.path.basename(path)} | missing={len(missing)} unexpected={len(unexpected)}")
    return model.eval()


@torch.no_grad()
def predict_all(models, dataset, device, batches):
    """
    Évalue chaque modèle avec la taille de batch de SON chiffre canonique
    (ResNet/ViT : bs=64 = éval d'entraînement ; DBFN : bs=1 = evaluate_hybrid_tta).
    Préprocess fait une seule fois. `batches` = dict {nom: batch_size}.
    """
    tf = preprocess()
    tensors, labels = [], []
    for i in range(len(dataset)):
        img, lab = dataset[i]
        tensors.append(tf(img)); labels.append(lab)
    labels = np.array(labels)

    preds = {}
    for k, m in models.items():
        bs = batches.get(k, 64)
        out = []
        for s in range(0, len(tensors), bs):
            x = torch.stack(tensors[s:s + bs]).to(device)
            out.append(m(x).argmax(1).cpu().numpy())
        preds[k] = np.concatenate(out)
    return preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/fer2013")
    ap.add_argument("--resnet_ckpt", default=RESNET_CKPT)
    ap.add_argument("--vit_ckpt", default=VIT_CKPT)
    ap.add_argument("--dbfn_ckpt", default=DBFN_CKPT)
    ap.add_argument("--out_dir", default="results_interpretability")
    ap.add_argument("--tag", default="")  # suffixe fichiers de sortie (ex: _rafdb)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Modèles ---
    print("[1/4] Chargement des 3 modèles...")
    resnet = create_resnet50(num_classes=7, pretrained=False, dropout_rate=0.3).to(device)
    load_state(resnet, args.resnet_ckpt, device)
    vit = ViTForFER("vit_base_patch16_224", num_classes=7, pretrained=False,
                    dropout=0.2, drop_path_rate=0.0).to(device)
    load_state(vit, args.vit_ckpt, device)
    dbfn = create_dbfn(fusion_type="concat", freeze_backbones=False, proj_dim=512,
                       dropout=0.2, drop_path_rate=0.0, num_classes=7).to(device)
    load_state(dbfn, args.dbfn_ckpt, device)

    # --- Prédictions ---
    print("[2/4] Inférence test set...")
    test_ds = FER2013Dataset(root_dir=args.data_dir, split="test", transform=None)
    preds, labels = predict_all({"resnet": resnet, "vit": vit, "dbfn": dbfn}, test_ds, device,
                                batches={"resnet": 64, "vit": 64, "dbfn": 1})
    N = len(labels)

    r_ok = preds["resnet"] == labels
    v_ok = preds["vit"] == labels
    d_ok = preds["dbfn"] == labels

    both = np.mean(r_ok & v_ok) * 100
    only_r = np.mean(r_ok & ~v_ok) * 100
    only_v = np.mean(~r_ok & v_ok) * 100
    neither = np.mean(~r_ok & ~v_ok) * 100

    # Sous-ensemble litigieux : exactement une branche a raison
    contested = (r_ok ^ v_ok)
    n_contested = contested.sum()
    dbfn_on_contested = np.mean(d_ok[contested]) * 100
    # borne : si DBFN choisissait toujours la bonne branche -> 100% ; oracle
    # référence : accuracy de chaque branche sur ce sous-ensemble
    r_on_contested = np.mean(r_ok[contested]) * 100
    v_on_contested = np.mean(v_ok[contested]) * 100

    print("\n" + "=" * 64)
    print("ANALYSE DE COMPLÉMENTARITÉ (test set, N=%d)" % N)
    print("=" * 64)
    print(f"  Accuracy ResNet-50 : {r_ok.mean()*100:.2f}%")
    print(f"  Accuracy ViT-Base  : {v_ok.mean()*100:.2f}%")
    print(f"  Accuracy DBFN      : {d_ok.mean()*100:.2f}%")
    print("-" * 64)
    print(f"  Les deux ✓            : {both:5.2f}%")
    print(f"  ResNet seul ✓         : {only_r:5.2f}%")
    print(f"  ViT seul ✓            : {only_v:5.2f}%")
    print(f"  Les deux ✗            : {neither:5.2f}%")
    print(f"  → Complémentaire (XOR): {only_r+only_v:5.2f}%  (N={n_contested})")
    print("-" * 64)
    print(f"  Sur le sous-ensemble litigieux (1 seule branche ✓) :")
    print(f"    ResNet ✓ : {r_on_contested:.1f}%  |  ViT ✓ : {v_on_contested:.1f}%  |  DBFN ✓ : {dbfn_on_contested:.1f}%")
    print(f"  Plafond théorique (oracle qui choisit toujours la bonne branche) : 100%")
    print("=" * 64)

    # --- Grad-CAM (réutilise le code validé) ---
    print("[3/4] Grad-CAM branche ResNet du DBFN...")
    for p in dbfn.cnn_branch.resnet.parameters():
        p.requires_grad_(True)
    cam = GradCAM(dbfn, dbfn.cnn_branch.resnet.layer4[-1])
    tf = preprocess()
    examples = []
    for c in range(7):
        for i in range(len(test_ds)):
            img, lab = test_ds[i]
            if lab != c:
                continue
            x = tf(img).unsqueeze(0).to(device)
            cmap, probs, pidx = cam(x)
            if pidx[0] == c:
                examples.append((denorm(tf(img)), cmap[0], c, probs[0, c]))
                break

    # --- Figure ---
    print("[4/4] Figure...")
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 7, height_ratios=[1.05, 1.0], hspace=0.35, wspace=0.3)

    # Panneau A : décomposition de l'accord (barres empilées sur tout le test)
    axA = fig.add_subplot(gs[0, 0:3])
    cats = ["Both\ncorrect", "ResNet\nonly", "ViT\nonly", "Both\nwrong"]
    vals = [both, only_r, only_v, neither]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    bars = axA.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10)
    dl = args.data_dir.lower()
    dsname = "RAF-DB" if "rafdb" in dl else ("AffectNet-7" if "affectnet" in dl else "FER2013")
    axA.set_ylabel("% of test samples", fontsize=11)
    axA.set_title(f"(a) Branch agreement on {dsname} test set", fontsize=11)
    axA.set_ylim(0, max(vals) + 8)
    axA.spines[["top", "right"]].set_visible(False)

    # Panneau B : qui tranche le sous-ensemble litigieux
    axB = fig.add_subplot(gs[0, 3:5])
    names = ["ResNet-50", "ViT-Base", "DBFN"]
    cvals = [r_on_contested, v_on_contested, dbfn_on_contested]
    cbars = axB.bar(names, cvals, color=["#1f77b4", "#ff7f0e", "#9467bd"],
                    edgecolor="black", linewidth=0.5)
    for b, v in zip(cbars, cvals):
        axB.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
    axB.axhline(100, ls="--", color="gray", lw=1)
    axB.text(2.0, 101.5, "oracle 100%", fontsize=8, color="gray", ha="center")
    axB.set_ylabel("Accuracy (%)", fontsize=11)
    axB.set_title(f"(b) On contested samples\n(exactly one branch correct, N={n_contested})", fontsize=10)
    axB.set_ylim(0, 112)
    axB.spines[["top", "right"]].set_visible(False)

    # Panneau texte : résumé chiffré
    axT = fig.add_subplot(gs[0, 5:7]); axT.axis("off")
    txt = (f"Test accuracy\n"
           f"  ResNet-50 : {r_ok.mean()*100:.2f}%\n"
           f"  ViT-Base  : {v_ok.mean()*100:.2f}%\n"
           f"  DBFN      : {d_ok.mean()*100:.2f}%\n\n"
           f"Complementary (XOR): {only_r+only_v:.1f}%\n"
           f"of test samples are solved\n"
           f"by exactly one branch.\n"
           f"DBFN recovers {dbfn_on_contested:.1f}% of them.")
    axT.text(0.0, 0.95, txt, va="top", ha="left", fontsize=11, family="monospace")

    # Ligne Grad-CAM
    for j, (img, cmap_img, c, conf) in enumerate(examples):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(img)
        ax.imshow(np.array(Image.fromarray((cmap_img*255).astype(np.uint8)).resize((224, 224))),
                  cmap="jet", alpha=0.45)
        ax.set_title(f"{FER2013_CLASSES[c]} ({conf*100:.0f}%)", fontsize=9)
        ax.axis("off")
    fig.text(0.5, 0.46, "(c) Grad-CAM on the ResNet-50 branch — local, expression-discriminative regions",
             ha="center", fontsize=11)

    out = os.path.join(out_dir, f"complementarity_dbfn{args.tag}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OUT] {out}")

    # sauvegarde chiffres
    np.savez(os.path.join(out_dir, f"complementarity{args.tag}.npz"),
             resnet_acc=r_ok.mean(), vit_acc=v_ok.mean(), dbfn_acc=d_ok.mean(),
             both=both, only_r=only_r, only_v=only_v, neither=neither,
             dbfn_on_contested=dbfn_on_contested, n_contested=n_contested)


if __name__ == "__main__":
    main()
