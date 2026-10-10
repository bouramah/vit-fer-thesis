#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_hybrid.py — Option A optimale (2 phases) pour DBFN (ResNet50 + ViT-Base) sur FER2013
========================================================================================

Phase A: Backbones gelés -> entraînement fusion uniquement (warmup)
Phase B: Unfreeze partiel contrôlé (ResNet layer4 / layer3+layer4 + derniers blocks ViT)
         -> LR backbone très petit, LR fusion plus grand

Anti-disk:
- Sauvegarde seulement le BEST par défaut (checkpoint léger possible)

Auteur: adapté pour Boss (UKAG IASD)
"""

import os
import sys
import argparse
import random
import time
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# ===== Imports projet (TES fichiers) =====
from models.hybrid import create_dbfn 
from utils.dataset import FER2013Dataset, get_train_transforms, get_val_transforms, FER2013_CLASSES
from utils.metrics import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    print_classification_report,
    plot_training_curves
)

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_device(device_str="auto"):
    if device_str == "auto":
        if torch.cuda.is_available():
            d = torch.device("cuda")
            print(f"[INFO] GPU NVIDIA: {torch.cuda.get_device_name(0)}")
            return d
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[INFO] GPU Apple (MPS)")
            return torch.device("mps")
        else:
            print("[INFO] CPU")
            return torch.device("cpu")
    return torch.device(device_str)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Data
# ----------------------------

def make_weighted_sampler_from_indices(full_train: FER2013Dataset, train_idx: np.ndarray):
    """
    WeightedRandomSampler basé sur la distribution des labels du train split.
    """
    train_labels = [full_train.samples[i][1] for i in train_idx]
    counts = np.bincount(np.array(train_labels), minlength=len(FER2013_CLASSES))
    weights = 1.0 / (counts.astype(np.float64) + 1e-6)
    sample_weights = weights[np.array(train_labels)]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    print("[SAMPLER] WeightedRandomSampler activé")
    for i, name in enumerate(FER2013_CLASSES):
        print(f"  {name:<10} count={counts[i]:5d} weight={weights[i]:.4f}")
    return sampler

def create_dataloaders(args):
    """
    Split stratifié (train/val) à partir de split="train"
    + test séparé split="test"
    """
    print("[DATA] Chargement dataset neutre (transform=None) pour split...")
    full_train = FER2013Dataset(root_dir=args.data_dir, split="train", transform=None)

    y = np.array([label for _, label in full_train.samples])
    # split_seed séparé: permet de FIGER le split train/val (celui des backbones)
    # tout en faisant varier le seed d'entraînement (init tête, ordre des batches).
    # Évite la fuite: avec un split différent, le "val" contiendrait des images vues
    # par les backbones pendant LEUR entraînement (split seed 42).
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=split_seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    print(f"[DATA] split_seed={split_seed} (train_seed={args.seed})")

    print("[DATA] Split stratifié:")
    print(f"  - Train: {len(train_idx)} images ({100*(1-args.val_ratio):.0f}%)")
    print(f"  - Val:   {len(val_idx)} images ({100*args.val_ratio:.0f}%)")

    train_tf = get_train_transforms(img_size=224, use_augmentation=True)
    val_tf   = get_val_transforms(img_size=224)

    train_ds_full = FER2013Dataset(root_dir=args.data_dir, split="train", transform=train_tf)
    val_ds_full   = FER2013Dataset(root_dir=args.data_dir, split="train", transform=val_tf)
    test_ds       = FER2013Dataset(root_dir=args.data_dir, split="test",  transform=val_tf)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds   = Subset(val_ds_full, val_idx)

    print(f"  - Test:  {len(test_ds)} images (évaluation finale)")

    sampler = None
    shuffle = True
    if args.weighted_sampler:
        sampler = make_weighted_sampler_from_indices(full_train, train_idx)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    return train_loader, val_loader, test_loader

# ----------------------------
# Mixup
# ----------------------------

def mixup_data(x, y, alpha=0.1):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------------
# Train / Val
# ----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_mixup=False, mixup_alpha=0.1):
    model.train()
    running_loss = 0.0

    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            # métriques train pas fiables sous mixup
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{running_loss/(pbar.n+1):.4f}"})

    epoch_loss = running_loss / len(loader)

    if use_mixup or len(all_preds) == 0:
        return epoch_loss, None, None

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100.0 * np.mean(all_preds == all_labels)
    f1  = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return epoch_loss, acc, f1

@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    val_loss = running_loss / len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = 100.0 * np.mean(all_preds == all_labels)
    f1  = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return val_loss, acc, f1, all_preds, all_labels

@torch.no_grad()
def evaluate_test(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    pbar = tqdm(loader, desc="Test", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        all_preds.extend(predicted.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100.0 * np.mean(all_preds == all_labels)
    f1  = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    test_loss = running_loss / len(loader)
    return test_loss, acc, f1, all_preds, all_labels

# ----------------------------
# Checkpoint (anti disque)
# ----------------------------

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, val_f1=None, save_optimizer=True):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_f1": val_f1,
    }
    if save_optimizer and optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if save_optimizer and scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(ckpt, path)

# ----------------------------
# Phase B : unfreeze partiel
# ----------------------------

def partial_unfreeze_dbfn(model, resnet_mode: str, vit_last_blocks: int):
    """
    Dégele uniquement certaines parties des backbones.
    - ResNet: layer4 ou layer3+layer4
    - ViT: derniers K blocks
    """
    # CNN branch = model.cnn_branch.resnet
    # ViT branch = model.vit_branch.vit (timm ViT)
    # Les noms exacts dépendent de ton models/hybrid.py.
    # Ici on suppose DBFN a : model.cnn_branch.resnet et model.vit_branch.vit

    # 1) ResNet
    if hasattr(model, "cnn_branch") and hasattr(model.cnn_branch, "resnet"):
        resnet = model.cnn_branch.resnet
        # par sécurité: on garde tout gelé d'abord
        for p in resnet.parameters():
            p.requires_grad = False

        to_unfreeze = []
        if resnet_mode == "layer4":
            to_unfreeze = ["layer4"]
        elif resnet_mode == "layer3_layer4":
            to_unfreeze = ["layer3", "layer4"]
        else:
            raise ValueError("resnet_unfreeze doit être: layer4 ou layer3_layer4")

        for name, module in resnet.named_children():
            if name in to_unfreeze:
                for p in module.parameters():
                    p.requires_grad = True

        print(f"[CNN] Unfreeze partiel: {to_unfreeze}")
    else:
        print("[CNN] ⚠️ cnn_branch.resnet introuvable, skip unfreeze CNN")

    # 2) ViT
    if hasattr(model, "vit_branch") and hasattr(model.vit_branch, "vit"):
        vit = model.vit_branch.vit
        for p in vit.parameters():
            p.requires_grad = False

        # timm ViT: vit.blocks est une liste de blocks transformer
        if hasattr(vit, "blocks"):
            blocks = vit.blocks
            k = int(vit_last_blocks)
            k = max(0, min(k, len(blocks)))
            if k > 0:
                for b in blocks[-k:]:
                    for p in b.parameters():
                        p.requires_grad = True
            print(f"[ViT] Unfreeze partiel: last {k} blocks")
        else:
            print("[ViT] ⚠️ vit.blocks introuvable, skip unfreeze ViT")
    else:
        print("[ViT] ⚠️ vit_branch.vit introuvable, skip unfreeze ViT")

# ----------------------------
# Optimizer groups
# ----------------------------

def build_optimizer(model, lr_backbone, lr_fusion, weight_decay):
    """
    LR différent:
    - fusion: lr_fusion
    - backbones (params requires_grad True hors fusion): lr_backbone
    """
    backbone_params, fusion_params = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "fusion" in name:
            fusion_params.append(p)
        else:
            backbone_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    groups.append({"params": fusion_params, "lr": lr_fusion, "weight_decay": weight_decay})

    opt = optim.AdamW(groups)
    return opt

def build_scheduler(optimizer, epochs, warmup_epochs, min_lr):
    warmup_epochs = int(max(0, min(warmup_epochs, epochs - 1)))
    if warmup_epochs == 0:
        cosine = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        return cosine

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
    sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return sched

# ----------------------------
# Main
# ----------------------------

def main(args):
    print("\n" + "=" * 70)
    print("🔀 ENTRAÎNEMENT DBFN — Option A optimale (Warmup Fusion + Partial Fine-tune)")
    print("=" * 70)

    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.results_dir)
    ensure_dir(args.log_dir)

    set_seed(args.seed)
    device = get_device(args.device)

    # Exp name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"dbfn_{args.fusion}_optA_{timestamp}"

    # Config print
    print("\n[CONFIG]")
    print(f"  Fusion: {args.fusion}")
    print(f"  Phase A freeze_backbones: {args.freeze_backbones}")
    print(f"  unfreeze_epoch: {args.unfreeze_epoch}")
    print(f"  ResNet unfreeze: {args.resnet_unfreeze}")
    print(f"  ViT last blocks: {args.vit_unfreeze_blocks}")
    print(f"  epochs={args.epochs} bs={args.batch_size}")
    print(f"  lr_backbone={args.lr_backbone} lr_fusion={args.lr_fusion}")
    print(f"  weight_decay={args.weight_decay} dropout={args.dropout}")
    print(f"  label_smoothing={args.label_smoothing}")
    print(f"  mixup={args.mixup} weighted_sampler={args.weighted_sampler}")
    print(f"  save_best_only={args.save_best_only} save_optimizer={args.save_optimizer}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(args)

    # Model
    print("\n[MODEL] Création DBFN...")
    model = create_dbfn(
        cnn_checkpoint=args.cnn_checkpoint,
        vit_checkpoint=args.vit_checkpoint,
        fusion_type=args.fusion,
        freeze_backbones=args.freeze_backbones,  # Phase A freeze
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        num_classes=7
    )
    model = model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Mixup règle: si sampler ON, mixup OFF
    use_mixup = (args.mixup > 0) and (not args.weighted_sampler)

    # Optimizer / Scheduler (Phase A)
    if args.freeze_backbones:
        # seuls params trainables sont la fusion
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr_fusion, weight_decay=args.weight_decay)
        print(f"[OPTIM] AdamW (fusion only) lr={args.lr_fusion:.2e}")
    else:
        optimizer = build_optimizer(model, args.lr_backbone, args.lr_fusion, args.weight_decay)
        print(f"[OPTIM] AdamW (groups) lr_backbone={args.lr_backbone:.2e} lr_fusion={args.lr_fusion:.2e}")

    warmup_epochs = min(args.warmup_epochs, max(0, args.epochs // 5))
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs, args.min_lr)

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, exp_name))

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()

    print("\n" + "=" * 70)
    print("🚀 DÉBUT ENTRAÎNEMENT")
    print("=" * 70 + "\n")

    phase_b_done = False

    for epoch in range(args.epochs):
        t0 = time.time()

        # Phase B trigger: unfreeze partiel + rebuild optimizer/scheduler
        if args.freeze_backbones and (not phase_b_done) and (args.unfreeze_epoch is not None) and (epoch + 1 == args.unfreeze_epoch):
            print("\n[UNFREEZE] Début Phase B: partial unfreeze + rebuild optimizer/scheduler")
            # on dégelé seulement une partie
            partial_unfreeze_dbfn(model, args.resnet_unfreeze, args.vit_unfreeze_blocks)

            optimizer = build_optimizer(model, args.lr_backbone, args.lr_fusion, args.weight_decay)
            scheduler = build_scheduler(optimizer, args.epochs - epoch, warmup_epochs=min(3, max(1, (args.epochs - epoch)//10)), min_lr=args.min_lr)
            phase_b_done = True
            print(f"[OPTIM] AdamW (groups) lr_backbone={args.lr_backbone:.2e} lr_fusion={args.lr_fusion:.2e}")

        # Train
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mixup=use_mixup, mixup_alpha=args.mixup
        )

        # Val
        va_loss, va_acc, va_f1, va_preds, va_labels = validate(
            model, val_loader, criterion, device, epoch
        )

        # Step scheduler (sans param epoch -> évite warning)
        scheduler.step()

        # logs
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_accs.append(va_acc)
        train_accs.append(float("nan") if tr_acc is None else tr_acc)

        epoch_s = time.time() - t0
        lr_show = optimizer.param_groups[-1]["lr"]
        gap = (tr_acc - va_acc) if (tr_acc is not None) else None

        if tr_acc is None:
            print(f"Epoch {epoch+1:3d}/{args.epochs} ({epoch_s:.1f}s) | "
                  f"Train Loss {tr_loss:.4f} (mixup ON) | "
                  f"Val Loss {va_loss:.4f} Acc {va_acc:.2f}% F1 {va_f1:.2f}% | LR {lr_show:.2e}")
        else:
            msg = (f"Epoch {epoch+1:3d}/{args.epochs} ({epoch_s:.1f}s) | "
                   f"Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% F1 {tr_f1:.2f}% | "
                   f"Val Loss {va_loss:.4f} Acc {va_acc:.2f}% F1 {va_f1:.2f}% | LR {lr_show:.2e}")
            if gap is not None:
                msg += f" | Gap {gap:.2f}%"
            print(msg)

        # TB
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Acc/val", va_acc, epoch)
        writer.add_scalar("F1/val", va_f1, epoch)
        writer.add_scalar("LR/fusion_group", lr_show, epoch)

        # Best logic
        is_best = va_f1 > best_val_f1
        if is_best:
            best_val_f1 = va_f1
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  ✓ Nouveau meilleur modèle! (Val F1: {va_f1:.2f}%)")
        else:
            patience_counter += 1

        # Save checkpoints (anti disque)
        if is_best or (not args.save_best_only and ((epoch + 1) % args.save_every == 0)):
            ckpt_name = f"{exp_name}_epoch{epoch+1}{'_best' if is_best else ''}.pth"
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
            save_checkpoint(
                ckpt_path, model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_f1=va_f1,
                save_optimizer=args.save_optimizer
            )

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n[EARLY STOP] Pas d'amélioration depuis {args.patience} epochs.")
            break

    total_min = (time.time() - start_time) / 60.0

    # Load best
    print("\n" + "=" * 70)
    print("📊 ÉVALUATION FINALE SUR TEST SET")
    print("=" * 70)

    # Retrouver le best ckpt (le dernier best enregistré)
    best_ckpt = None
    for fn in os.listdir(args.checkpoint_dir):
        if fn.startswith(exp_name) and fn.endswith("_best.pth"):
            best_ckpt = os.path.join(args.checkpoint_dir, fn)

    if best_ckpt and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Best loaded: epoch {ckpt.get('epoch', 'N/A')+1 if isinstance(ckpt.get('epoch', None), int) else ckpt.get('epoch','N/A')}, Val F1: {ckpt.get('val_f1','N/A')}")
    else:
        print("[WARN] Aucun checkpoint best trouvé, utilisation du modèle courant.")

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate_test(model, test_loader, criterion, device)

    print("\n🎯 TEST RESULTS:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Macro-F1: {test_f1:.2f}%")

    print_classification_report(test_labels, test_preds, f"DBFN_{args.fusion}_OptA")

    # Figures
    fig_prefix = f"{exp_name}"
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=os.path.join(args.results_dir, f"{fig_prefix}_confusion_matrix.png"),
        title=f"Matrice de Confusion - DBFN {args.fusion} OptA"
    )
    plot_per_class_accuracy(
        test_labels, test_preds,
        save_path=os.path.join(args.results_dir, f"{fig_prefix}_per_class_accuracy.png"),
        title=f"Accuracy par Classe - DBFN {args.fusion} OptA"
    )
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(args.results_dir, f"{fig_prefix}_training_curves.png")
    )

    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"  Exp: {exp_name}")
    print(f"  Best Val F1: {best_val_f1:.2f}% (epoch {best_epoch})")
    print(f"  Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.2f}%")
    print(f"  Temps total: {total_min:.1f} min")
    print("=" * 70 + "\n")

    writer.close()


# ----------------------------
# Args
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DBFN Option A optimale (fusion warmup + partial fine-tune)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    p.add_argument("--data_dir", type=str, default="data/fer2013")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--val_ratio", type=float, default=0.1)

    # Checkpoints
    p.add_argument("--cnn_checkpoint", type=str, required=True)
    p.add_argument("--vit_checkpoint", type=str, required=True)

    # Model
    p.add_argument("--fusion", type=str, default="concat", choices=["concat", "cross_attention", "gating"])
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--drop_path_rate", type=float, default=0.1)

    # Option A (2 phases)
    p.add_argument("--freeze_backbones", action="store_true", help="Phase A: geler backbones au début")
    p.add_argument("--unfreeze_epoch", type=int, default=10, help="Epoch où démarre la Phase B (partial unfreeze)")
    p.add_argument("--resnet_unfreeze", type=str, default="layer4", choices=["layer4", "layer3_layer4"])
    p.add_argument("--vit_unfreeze_blocks", type=int, default=2, help="Nombre de blocks ViT (derniers) à dégeler en Phase B")

    # Training
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--lr_backbone", type=float, default=1e-6)
    p.add_argument("--lr_fusion", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--weighted_sampler", action="store_true", default=False)

    # Scheduler
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-7)

    # Early stop + save
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--save_best_only", action="store_true", default=True, help="Sauvegarder seulement les checkpoints best")
    p.add_argument("--save_optimizer", action="store_true", default=False, help="Inclure optimizer/scheduler dans checkpoint (plus lourd)")

    # Output
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_hybrid_optA")
    p.add_argument("--results_dir", type=str, default="results_hybrid_optA")
    p.add_argument("--log_dir", type=str, default="logs_hybrid_optA")

    # Misc
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=None,
                   help="Seed du split train/val (figer au seed des backbones=42). Si None -> = --seed")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.data_dir):
        print(f"❌ Dataset non trouvé: {args.data_dir}")
        sys.exit(1)

    main(args)