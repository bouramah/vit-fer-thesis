#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
attention_compare.py — POURQUOI pas de complémentarité SPATIALE ? (investigation)
=================================================================================
Compare la focalisation spatiale des deux branches :
  - Grad-CAM sur ResNet-50 (layer4)
  - Attention CLS->patches du ViT-Base (dernière couche, moyenne des têtes)
sur les mêmes images, en distinguant :
  - images où les 2 branches sont correctes (accord)
  - images "litigieuses" (exactement une branche correcte)

Quantifie le recouvrement spatial (corrélation de Pearson + IoU du top-25%)
pour tester si les cartes DIFFÈRENT là où les branches sont en désaccord.

Usage: CUDA_VISIBLE_DEVICES=2 venv_fer/bin/python attention_compare.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.resnet import create_resnet50
from models.vit import ViTForFER
from utils.dataset import FER2013Dataset, FER2013_CLASSES
from analyze_dbfn import GradCAM, denorm

MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
RESNET_CKPT = "checkpoints_run3_resnet50/resnet50_v2_epoch61_best.pth"
VIT_CKPT = "checkpoints_vit_base_v3_run1/vit_base_v3_epoch86_best.pth"


def tf(): return transforms.Compose([
    transforms.Resize((224, 224)), transforms.Grayscale(3),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    st = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(st, strict=False)
    return model.eval()


def vit_attention(vit, x):
    """Carte d'attention CLS->patches, dernière couche, moyenne des têtes -> 14x14 normalisée."""
    blk = vit.backbone.blocks[-1].attn
    blk.fused_attn = False
    store = {}
    def hook(mod, inp):
        t = inp[0]; B, N, C = t.shape
        qkv = mod.qkv(t).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        a = (q @ k.transpose(-2, -1)) * mod.scale
        store['a'] = a.softmax(-1).detach()
    h = blk.register_forward_pre_hook(hook)
    with torch.no_grad():
        vit(x)
    h.remove()
    a = store['a'][0, :, 0, 1:].mean(0).reshape(14, 14)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a.cpu().numpy()


def to_grid(m, size=14):
    """Redimensionne une carte 2D en size×size (pour comparer CAM 7x7 et attn 14x14)."""
    im = Image.fromarray((m * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    g = np.asarray(im, dtype=np.float32) / 255.0
    return (g - g.min()) / (g.max() - g.min() + 1e-8)


def overlap(cam, attn):
    """Recouvrement spatial : corrélation de Pearson + IoU du top-25%."""
    c = to_grid(cam).flatten(); a = to_grid(attn).flatten()
    corr = np.corrcoef(c, a)[0, 1] if c.std() > 1e-6 and a.std() > 1e-6 else 0.0
    tc = c >= np.quantile(c, 0.75); ta = a >= np.quantile(a, 0.75)
    iou = (tc & ta).sum() / ((tc | ta).sum() + 1e-8)
    return corr, iou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "results_interpretability"; os.makedirs(out_dir, exist_ok=True)

    print("[1/4] Modèles...")
    resnet = load(create_resnet50(num_classes=7, pretrained=False, dropout_rate=0.3).to(device), RESNET_CKPT, device)
    vit = load(ViTForFER("vit_base_patch16_224", num_classes=7, pretrained=False, dropout=0.2,
                         drop_path_rate=0.0).to(device), VIT_CKPT, device)

    ds = FER2013Dataset(root_dir="data/fer2013", split="test", transform=None)
    T = tf()

    print("[2/4] Prédictions (catégorisation accord / litige)...")
    rp, vp, labs = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, lab = ds[i]
            x = T(img).unsqueeze(0).to(device)
            rp.append(resnet(x).argmax(1).item()); vp.append(vit(x).argmax(1).item()); labs.append(lab)
    rp, vp, labs = np.array(rp), np.array(vp), np.array(labs)
    r_ok, v_ok = rp == labs, vp == labs
    idx_both = np.where(r_ok & v_ok)[0]
    idx_contested = np.where(r_ok ^ v_ok)[0]
    print(f"  both-correct={len(idx_both)}  contested={len(idx_contested)}")

    print("[3/4] Recouvrement spatial CAM(ResNet) vs Attn(ViT)...")
    cam = GradCAM(resnet, resnet.backbone.layer4[-1])
    rng = np.random.RandomState(42)

    def stats(indices, n=200):
        sel = indices if len(indices) <= n else rng.choice(indices, n, replace=False)
        cs, ios = [], []
        for i in sel:
            img, _ = ds[i]
            x = T(img).unsqueeze(0).to(device)
            cmap, _, _ = cam(x)          # 7x7 (grad requis)
            amap = vit_attention(vit, x) # 14x14
            c, io = overlap(cmap[0], amap)
            cs.append(c); ios.append(io)
        return np.mean(cs), np.mean(ios), len(sel)

    corr_b, iou_b, nb = stats(idx_both)
    corr_c, iou_c, nc = stats(idx_contested)
    # baseline aléatoire d'IoU pour top-25% sur grille 14x14 ~ 0.25*0.25/(0.4375) ≈ 0.143
    print("=" * 60)
    print("RECOUVREMENT SPATIAL (corrélation Pearson / IoU top-25%)")
    print("=" * 60)
    print(f"  Accord (both correct, n={nb})   : corr={corr_b:.3f}  IoU={iou_b:.3f}")
    print(f"  Litige (contested, n={nc})       : corr={corr_c:.3f}  IoU={iou_c:.3f}")
    print(f"  (IoU aléatoire attendu ≈ 0.14)")
    print("=" * 60)

    print("[4/4] Figure exemples (orig | ResNet Grad-CAM | ViT attention)...")
    def pick(indices, k=2):
        return list(rng.choice(indices, min(k, len(indices)), replace=False))
    rows = [("both", i) for i in pick(idx_both, 2)] \
         + [("R✓ V✗", i) for i in pick(np.where(r_ok & ~v_ok)[0], 2)] \
         + [("V✓ R✗", i) for i in pick(np.where(v_ok & ~r_ok)[0], 2)]

    fig, axes = plt.subplots(len(rows), 3, figsize=(7, 2.3 * len(rows)))
    for r, (tag, i) in enumerate(rows):
        img, lab = ds[i]
        x = T(img).unsqueeze(0).to(device)
        cmap, _, _ = cam(x); amap = vit_attention(vit, x)
        base = denorm(T(img))
        axes[r, 0].imshow(base); axes[r, 0].set_ylabel(f"{tag}\n{FER2013_CLASSES[lab]}", fontsize=8)
        axes[r, 1].imshow(base)
        axes[r, 1].imshow(np.array(Image.fromarray((cmap[0]*255).astype(np.uint8)).resize((224,224))), cmap="jet", alpha=0.45)
        axes[r, 2].imshow(base)
        axes[r, 2].imshow(np.array(Image.fromarray((amap*255).astype(np.uint8)).resize((224,224))), cmap="jet", alpha=0.45)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_title("Input", fontsize=10)
    axes[0, 1].set_title("ResNet-50 Grad-CAM", fontsize=10)
    axes[0, 2].set_title("ViT-Base attention", fontsize=10)
    plt.tight_layout()
    out = os.path.join(out_dir, "attention_compare.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OUT] {out}")
    np.savez(os.path.join(out_dir, "spatial_overlap.npz"),
             corr_both=corr_b, iou_both=iou_b, corr_contested=corr_c, iou_contested=iou_c)


if __name__ == "__main__":
    main()
