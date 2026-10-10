#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_complexity.py — Métriques de complexité DBFN pour CARI 2026
=====================================================================
Répond aux reviewers #10 (FLOPs/MACs, taille Mo, params) et #11 (coût TTA).

Mesure, pour DBFN-concat, DBFN-cross_attn, ResNet-50 seul, ViT-Base seul :
  - Paramètres : total + entraînables (config gelée) + détail par composant
  - FLOPs / MACs : fvcore (MACs, standard CV) + torch FlopCounterMode (FLOPs, gère l'attention)
  - Taille mémoire du modèle (Mo, fp32)
  - Latence (ms) et débit (FPS) GPU @ batch=1 et batch=64
  - Coût TTA : latence/FPS effectifs pour none(1), hflip(2), 5crop(5), 10crop(10)

Usage:
  CUDA_VISIBLE_DEVICES=2 venv_fer/bin/python benchmark_complexity.py
"""

import time
import json
import argparse
import contextlib

import torch
import torch.nn as nn

from models.hybrid import create_dbfn, ResNet50Branch, ViTBaseBranch


# ----------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------

def count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(module: nn.Module) -> float:
    """Taille des poids en Mo (fp32 = 4 octets/param)."""
    n_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in module.buffers())
    return n_bytes / (1024 ** 2)


# ----------------------------------------------------------------------
# FLOPs / MACs
# ----------------------------------------------------------------------

def flops_fvcore(model, x):
    """MACs via fvcore (ce que la plupart des papiers CV appellent 'FLOPs')."""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return fca.total()  # MACs
    except Exception as e:
        print(f"  [fvcore] échec: {e}")
        return None


def flops_torch(model, x):
    """FLOPs via le compteur intégré torch (gère scaled_dot_product_attention)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        fcm = FlopCounterMode(display=False)
        with fcm:
            model(x)
        return fcm.get_total_flops()  # FLOPs
    except Exception as e:
        print(f"  [torch] échec: {e}")
        return None


# ----------------------------------------------------------------------
# Latence / débit
# ----------------------------------------------------------------------

@torch.no_grad()
def benchmark_latency(model, device, batch_size=1, n_warmup=10, n_iter=50, passes=1):
    """
    Latence moyenne (ms) et FPS pour `passes` forward par échantillon (TTA).
    passes>1 simule le coût TTA : on traite passes*batch_size images par étape.
    """
    model.eval()
    x = torch.randn(batch_size * passes, 3, 224, 224, device=device)

    is_cuda = device.type == "cuda"
    for _ in range(n_warmup):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # temps par "image logique" (batch_size images, chacune vue `passes` fois)
    ms_per_batch = (elapsed / n_iter) * 1000.0
    imgs = batch_size  # nb d'images logiques par étape
    fps = (imgs * n_iter) / elapsed
    return ms_per_batch, fps


# ----------------------------------------------------------------------
# Construction des modèles
# ----------------------------------------------------------------------

def build_models(device):
    """Construit les 4 modèles SANS charger de checkpoints FER (complexité = indépendante des poids)."""
    models = {}

    # DBFN concat (config gelée du papier)
    m = create_dbfn(cnn_checkpoint=None, vit_checkpoint=None,
                    fusion_type="concat", freeze_backbones=True,
                    proj_dim=512, dropout=0.2, drop_path_rate=0.0, num_classes=7)
    models["DBFN-concat"] = m.to(device).eval()

    # DBFN cross-attention
    m = create_dbfn(cnn_checkpoint=None, vit_checkpoint=None,
                    fusion_type="cross_attention", freeze_backbones=True,
                    proj_dim=512, dropout=0.2, drop_path_rate=0.0, num_classes=7)
    models["DBFN-cross_attn"] = m.to(device).eval()

    # ResNet-50 seul (branche + tête linéaire 7 classes pour être comparable)
    class ResNet50Cls(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch = ResNet50Branch(pretrained_checkpoint=None, freeze=False)
            self.head = nn.Linear(self.branch.feature_dim, 7)
        def forward(self, x):
            return self.head(self.branch(x))
    models["ResNet-50"] = ResNet50Cls().to(device).eval()

    # ViT-Base seul
    class ViTBaseCls(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch = ViTBaseBranch(pretrained_checkpoint=None, freeze=False, drop_path_rate=0.0)
            self.head = nn.Linear(self.branch.feature_dim, 7)
        def forward(self, x):
            return self.head(self.branch(x))
    models["ViT-Base"] = ViTBaseCls().to(device).eval()

    return models


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_iter", type=int, default=50)
    ap.add_argument("--out", default="results_complexity/complexity_metrics.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print("[INFO] Construction des modèles (sans checkpoints FER)...\n")

    models = build_models(device)
    x1 = torch.randn(1, 3, 224, 224, device=device)

    results = {}

    # Latence DBFN-concat @ bs1 mesurée UNE SEULE FOIS -> source de vérité unique
    # (réutilisée pour le tableau #10 ET la ligne "None" du tableau #11).
    dbfn = models["DBFN-concat"]
    dbfn_lat_bs1, _ = benchmark_latency(dbfn, device, batch_size=1,
                                        n_warmup=20, n_iter=args.n_iter)

    print("=" * 78)
    print("TABLEAU #10 — COMPLEXITÉ (params, GFLOPs, taille, latence, FPS)")
    print("=" * 78)
    header = f"{'Modèle':18} {'Total(M)':>9} {'Train(M)':>9} {'GFLOPs':>8} {'Mo':>7} {'ms@bs1':>8} {'FPS@bs64':>9}"
    print(header)
    print("-" * 78)

    for name, model in models.items():
        total, trainable = count_params(model)
        size = model_size_mb(model)
        flops = flops_torch(model, x1)  # FLOPs (torch, gère l'attention) -> chiffre unique
        if name == "DBFN-concat":
            lat1 = dbfn_lat_bs1  # même mesure que #11 "None"
        else:
            lat1, _ = benchmark_latency(model, device, batch_size=1, n_warmup=20, n_iter=args.n_iter)
        _, fps64 = benchmark_latency(model, device, batch_size=64, n_warmup=10, n_iter=max(10, args.n_iter // 2))

        results[name] = {
            "params_total_M": total / 1e6,
            "params_trainable_M": trainable / 1e6,
            "gflops": flops / 1e9 if flops else None,
            "size_mb": size,
            "latency_ms_bs1": lat1,
            "fps_bs64": fps64,
        }
        gflops_s = f"{flops/1e9:8.2f}" if flops else "     n/a"
        print(f"{name:18} {total/1e6:9.2f} {trainable/1e6:9.2f} {gflops_s} "
              f"{size:7.1f} {lat1:8.2f} {fps64:9.1f}")

    # ------------------------------------------------------------------
    # TTA : coût pour le DBFN-concat (modèle retenu)
    # "None" réutilise EXACTEMENT la latence bs1 du tableau #10 (cohérence).
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLEAU #11 — COÛT DU TTA (DBFN-concat, batch logique = 1 image)")
    print("=" * 78)
    print(f"{'TTA':10} {'Passes':>7} {'ms/image':>10} {'FPS':>9}")
    print("-" * 78)

    tta_passes = {"none": 1, "hflip": 2, "5crop": 5, "10crop": 10}
    results["TTA_cost_DBFN_concat"] = {}
    for tta, passes in tta_passes.items():
        if passes == 1:
            ms = dbfn_lat_bs1                 # source de vérité unique
            fps = 1000.0 / ms
        else:
            ms, fps = benchmark_latency(dbfn, device, batch_size=1,
                                        n_warmup=20, n_iter=args.n_iter, passes=passes)
        results["TTA_cost_DBFN_concat"][tta] = {"passes": passes, "ms_per_image": ms, "fps": fps}
        print(f"{tta:10} {passes:7d} {ms:10.2f} {fps:9.1f}")

    # Sauvegarde
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUT] Métriques sauvegardées: {args.out}")


if __name__ == "__main__":
    main()
