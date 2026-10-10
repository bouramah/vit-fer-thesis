#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_dbfn.py — Interprétabilité DBFN pour CARI 2026 (items #14a, #14b, #17)
=============================================================================
Produit :
  - #14a : t-SNE des features fusionnées (et ResNet-seul / ViT-seul) colorées par classe
           -> montre la complémentarité / séparabilité (R2 "feature complementarity").
  - #14b : Grad-CAM sur la branche ResNet-50 (layer4) pour quelques exemples par émotion.
  - #9   : rapport Précision/Rappel/F1 par classe (réutilise utils.metrics).

Figure finale = panneau multi-vues (t-SNE + grille Grad-CAM) -> 1 seule figure pour le papier.

Usage:
  CUDA_VISIBLE_DEVICES=2 venv_fer/bin/python analyze_dbfn.py \
      --checkpoint checkpoints_ablation_concat/<best>.pth --fusion concat \
      --out_dir results_interpretability --tsne_n 2000
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms
from models.hybrid import create_dbfn
from utils.dataset import FER2013Dataset, FER2013_CLASSES
from utils.metrics import print_classification_report

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def val_preprocess(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(checkpoint, fusion, device, proj_dim=512):
    model = create_dbfn(cnn_checkpoint=None, vit_checkpoint=None, fusion_type=fusion,
                        freeze_backbones=False, proj_dim=proj_dim, dropout=0.2,
                        drop_path_rate=0.0, num_classes=7).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] {checkpoint} | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


# ----------------------------------------------------------------------
# Collecte des features (branches + prédictions)
# ----------------------------------------------------------------------

@torch.no_grad()
def collect_features(model, dataset, device, max_n=2000, batch=64):
    """Récupère features CNN (2048), ViT (768), concat (2816), préds et labels."""
    tf = val_preprocess()
    idx = np.arange(len(dataset))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    idx = idx[:max_n] if max_n else idx

    cnn_f, vit_f, preds, labels = [], [], [], []
    buf, lab_buf = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        cf = model.cnn_branch(x)
        vf = model.vit_branch(x)
        logits = model.fusion(cf, vf)
        cnn_f.append(cf.cpu().numpy())
        vit_f.append(vf.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        labels.extend(lab_buf)
        buf.clear(); lab_buf.clear()

    for i in idx:
        img, lab = dataset[i]
        if not isinstance(img, Image.Image):
            # dataset peut renvoyer un tenseur si transform!=None; ici transform=None -> PIL
            pass
        buf.append(tf(img)); lab_buf.append(lab)
        if len(buf) == batch:
            flush()
    flush()

    return (np.concatenate(cnn_f), np.concatenate(vit_f),
            np.concatenate(preds), np.array(labels))


def run_tsne(feats, perplexity=30, seed=42):
    from sklearn.manifold import TSNE
    ts = TSNE(n_components=2, perplexity=perplexity, init="pca",
              learning_rate="auto", random_state=seed)
    return ts.fit_transform(feats)


# ----------------------------------------------------------------------
# Grad-CAM sur la branche ResNet-50 (layer4)
# ----------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()  # retropropagation
        # poids = moyenne globale des gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [B,C,1,1]
        cam = F.relu((weights * self.activations).sum(dim=1))      # [B,H,W]
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.cpu().numpy(), logits.softmax(1).detach().cpu().numpy(), class_idx.cpu().numpy()


def denorm(t):
    x = t.clone().cpu()
    for c in range(3):
        x[c] = x[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fusion", default="concat", choices=["concat", "cross_attention", "gating"])
    ap.add_argument("--data_dir", default="data/fer2013")
    ap.add_argument("--out_dir", default="results_interpretability")
    ap.add_argument("--tsne_n", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.fusion, device)

    test_ds = FER2013Dataset(root_dir=args.data_dir, split="test", transform=None)

    # ---- Features + t-SNE (#14a) ----
    print("[1/3] Extraction features + t-SNE...")
    cnn_f, vit_f, preds, labels = collect_features(model, test_ds, device, max_n=args.tsne_n)
    np.savez(os.path.join(args.out_dir, "features.npz"),
             cnn=cnn_f, vit=vit_f, preds=preds, labels=labels)

    fused = np.concatenate([cnn_f, vit_f], axis=1)
    emb_fused = run_tsne(fused)
    emb_cnn = run_tsne(cnn_f)
    emb_vit = run_tsne(vit_f)

    # ---- Grad-CAM (#14b) : un exemple bien classé par classe ----
    print("[2/3] Grad-CAM branche ResNet...")
    for p in model.cnn_branch.resnet.parameters():
        p.requires_grad_(True)
    cam = GradCAM(model, model.cnn_branch.resnet.layer4[-1])
    tf = val_preprocess()
    examples = []
    for c in range(7):
        # premier test bien classé de la classe c
        found = None
        for i in range(len(test_ds)):
            img, lab = test_ds[i]
            if lab != c:
                continue
            x = tf(img).unsqueeze(0).to(device)
            # class_idx=None -> Grad-CAM sur la classe PRÉDITE (argmax)
            cmap, probs, pidx = cam(x)
            if pidx[0] == c:  # vrai positif : prédiction == vérité
                found = (denorm(tf(img)), cmap[0], c, probs[0, c])
                break
        if found:
            examples.append(found)

    # ---- Figure combinée (#14) : grille alignée 2×21 (3×7 t-SNE / 7×3 Grad-CAM) ----
    print("[3/3] Figure...")
    cmap_cls = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 21, hspace=0.25, wspace=0.3)

    for k, (emb, title) in enumerate([(emb_cnn, "ResNet-50 features"),
                                      (emb_vit, "ViT-Base features"),
                                      (emb_fused, "Fused (concat) features")]):
        ax = fig.add_subplot(gs[0, k * 7:(k + 1) * 7])
        for c in range(7):
            m = labels == c
            ax.scatter(emb[m, 0], emb[m, 1], s=5, color=cmap_cls(c), label=FER2013_CLASSES[c])
        ax.set_title(f"t-SNE — {title}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # légende compacte
    handles = [plt.Line2D([0], [0], marker='o', ls='', color=cmap_cls(c), label=FER2013_CLASSES[c]) for c in range(7)]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, 0.52))

    for j, (img, cmap_img, c, conf) in enumerate(examples):
        ax = fig.add_subplot(gs[1, j * 3:(j + 1) * 3])
        ax.imshow(img)
        ax.imshow(np.array(Image.fromarray((cmap_img * 255).astype(np.uint8)).resize((224, 224))),
                  cmap="jet", alpha=0.45)
        ax.set_title(f"{FER2013_CLASSES[c]} ({conf*100:.0f}%)", fontsize=9)
        ax.axis("off")

    out = os.path.join(args.out_dir, "interpretability_dbfn.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OUT] {out}")

    # ---- #9 rapport par classe ----
    print_classification_report(labels, preds, f"DBFN_{args.fusion}_subset")


if __name__ == "__main__":
    main()
