#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train DBFN-TOKENS (ResNet50 tokens + ViT tokens) pour FER2013
=============================================================
- Dataset: utils.dataset.FER2013Dataset (structure data/fer2013/train|test/<class>/*.jpg|png)
- Split: StratifiedShuffleSplit (train/val) sur le train set uniquement
- Backbones: ResNet-50 + ViT-Base (chargés depuis tes checkpoints FER)
- Fusion: Cross-attention bidirectionnelle au niveau tokens
- Training:
  * Freeze backbones (phase A) puis Option 2: Unfreeze progressif (phase B)
  * AdamW + Warmup (LinearLR) + CosineAnnealing
  * Label smoothing + (option) mixup + (option) weighted sampler
  * Early stopping sur Val Macro-F1

Auteur: Adapté pour Ibrahima Douty OULARE
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

import timm
from torchvision import models

# Imports locaux
from utils.dataset import (
    FER2013Dataset,
    get_train_transforms,
    get_val_transforms,
    FER2013_CLASSES
)
from utils.metrics import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    print_classification_report,
    plot_training_curves
)


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] GPU NVIDIA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[INFO] GPU Apple (MPS)")
        else:
            device = torch.device("cpu")
            print("[INFO] CPU")
    else:
        device = torch.device(device_str)
        print(f"[INFO] Device: {device}")
    return device


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# CHECKPOINT LOADING HELPERS
# ============================================================

def load_resnet50_backbone_from_ckpt(resnet: nn.Module, ckpt_path: str):
    """
    Charge ton checkpoint ResNet50 FER (train_v2) dans un resnet torchvision
    dont fc = Identity (donc on ignore fc).
    """
    print(f"[CNN Tokens] Chargement checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    filtered = {}
    for k, v in state.items():
        # Ignore classifier/fc final (7 classes)
        if "classifier" in k:
            continue
        if k.startswith("fc.") and v.ndim >= 1 and (v.shape[0] == 7):
            continue
        filtered[k] = v

    missing, unexpected = resnet.load_state_dict(filtered, strict=False)
    # On s'attend souvent à missing sur fc.* (puisqu'on a Identity)
    print(f"  Chargé tmp ResNet! Missing={len(missing)} Unexpected={len(unexpected)}")


def load_vit_backbone_from_ckpt(vit: nn.Module, ckpt_path: str):
    """
    Charge ton checkpoint ViTForFER (train_vit_v3) dans un timm vit num_classes=0
    en ne gardant que backbone.*.
    """
    print(f"[ViT Tokens] Chargement checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    filtered = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            filtered[k.replace("backbone.", "")] = v
        elif k.startswith("classifier") or k.startswith("head"):
            continue

    if len(filtered) == 0:
        # fallback
        filtered = {k: v for k, v in state.items()
                    if ("classifier" not in k and "head" not in k and not k.startswith("fc."))}

    missing, unexpected = vit.load_state_dict(filtered, strict=False)
    print(f"  Chargé! Missing={len(missing)} Unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing keys (ex 5): {missing[:5]}")


# ============================================================
# MODEL: TOKENS BRANCHES
# ============================================================

class ResNet50TokenBranch(nn.Module):
    """
    Sort des tokens CNN: features map layer4 -> (B, 2048, 7, 7) sur 224x224,
    converti en tokens (B, 49, 2048).
    """
    def __init__(self, ckpt_path: str = None, freeze: bool = False):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.feature_dim = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()

        if ckpt_path:
            load_resnet50_backbone_from_ckpt(self.resnet, ckpt_path)

        if freeze:
            self.freeze_all()

    def freeze_all(self):
        for p in self.resnet.parameters():
            p.requires_grad = False
        print("[CNN Tokens] Backbone gelé")

    def unfreeze_all(self):
        for p in self.resnet.parameters():
            p.requires_grad = True
        print("[CNN Tokens] Backbone dégelé")

    def unfreeze_last_layers(self, num_stages: int = 1):
        """
        Dégèle la fin de ResNet: layer4 (1), layer4+layer3 (2), etc.
        """
        stages = ["layer1", "layer2", "layer3", "layer4"]
        num_stages = max(1, min(4, num_stages))
        to_unfreeze = set(stages[-num_stages:])

        # Par défaut: tout gelé
        for p in self.resnet.parameters():
            p.requires_grad = False

        for name, module in self.resnet.named_children():
            if name in to_unfreeze:
                for p in module.parameters():
                    p.requires_grad = True

        # Toujours laisser BN en mode trainable ? ici on laisse comme params (requires_grad True)
        print(f"[CNN Tokens] Unfreeze partiel: {sorted(list(to_unfreeze))}")

    def forward(self, x):
        # Copie du forward torchvision resnet, mais on récupère feature map avant avgpool
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # (B, 2048, 7, 7)

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C) = (B, 49, 2048)
        return tokens


class ViTTokenBranch(nn.Module):
    """
    Sort des tokens ViT: forward_features -> (B, 1+N, 768) (CLS + patch tokens)
    """
    def __init__(self, ckpt_path: str = None, freeze: bool = False, drop_path_rate: float = 0.1):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate
        )
        self.feature_dim = self.vit.num_features  # 768

        if ckpt_path:
            load_vit_backbone_from_ckpt(self.vit, ckpt_path)
        else:
            print("[ViT Tokens] Aucun ckpt fourni -> ImageNet pretrained")
            vit_im = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0, drop_path_rate=drop_path_rate)
            self.vit.load_state_dict(vit_im.state_dict(), strict=True)
            del vit_im

        if freeze:
            self.freeze_all()

    def freeze_all(self):
        for p in self.vit.parameters():
            p.requires_grad = False
        print("[ViT Tokens] Backbone gelé")

    def unfreeze_all(self):
        for p in self.vit.parameters():
            p.requires_grad = True
        print("[ViT Tokens] Backbone dégelé")

    def unfreeze_last_blocks(self, num_blocks: int = 2):
        """
        Dégèle les derniers blocks du Transformer.
        vit.blocks est une liste de blocks.
        """
        blocks = getattr(self.vit, "blocks", None)
        if blocks is None:
            # fallback: tout dégelé
            self.unfreeze_all()
            return

        num_blocks = max(1, min(len(blocks), num_blocks))

        # tout gelé
        for p in self.vit.parameters():
            p.requires_grad = False

        # dégel last blocks
        for b in list(blocks)[-num_blocks:]:
            for p in b.parameters():
                p.requires_grad = True

        # on laisse LayerNorm final / head_norm si présent (souvent utile)
        for name in ["norm", "fc_norm", "head_norm"]:
            mod = getattr(self.vit, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

        print(f"[ViT Tokens] Unfreeze partiel: last {num_blocks} blocks")

    def forward(self, x):
        # timm vit forward_features renvoie tokens (B, 1+N, C)
        tokens = self.vit.forward_features(x)
        return tokens


# ============================================================
# FUSION TOKENS: Cross-Attention bidirectionnelle
# ============================================================

class TokenCrossFusion(nn.Module):
    def __init__(self, cnn_dim=2048, vit_dim=768, d_model=512, num_heads=8, dropout=0.2, num_classes=7):
        super().__init__()
        self.d_model = d_model

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.attn_cnn_q_vit_kv = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn_vit_q_cnn_kv = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, cnn_tokens, vit_tokens):
        """
        cnn_tokens: (B, 49, 2048)
        vit_tokens: (B, 1+196, 768) => on gardera CLS + patch tokens
        """
        cnn = self.cnn_proj(cnn_tokens)   # (B, 49, d)
        vit = self.vit_proj(vit_tokens)   # (B, 197, d)

        # CrossAttn 1: CNN query, ViT key/value
        cnn_att, _ = self.attn_cnn_q_vit_kv(query=cnn, key=vit, value=vit)
        cnn_fused = self.norm1(cnn + cnn_att)  # (B, 49, d)

        # CrossAttn 2: ViT query, CNN key/value
        vit_att, _ = self.attn_vit_q_cnn_kv(query=vit, key=cnn, value=cnn)
        vit_fused = self.norm2(vit + vit_att)  # (B, 197, d)

        # Pooling: mean sur tokens CNN + CLS token côté ViT
        cnn_pool = cnn_fused.mean(dim=1)       # (B, d)
        vit_cls  = vit_fused[:, 0, :]          # (B, d)

        fused = torch.cat([cnn_pool, vit_cls], dim=-1)  # (B, 2d)
        logits = self.ffn(fused)
        return logits


# ============================================================
# MODEL: DBFN TOKENS
# ============================================================

class DBFNTokens(nn.Module):
    def __init__(
        self,
        cnn_ckpt: str,
        vit_ckpt: str,
        freeze_backbones: bool,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        num_classes: int = 7
    ):
        super().__init__()

        self.cnn = ResNet50TokenBranch(ckpt_path=cnn_ckpt, freeze=freeze_backbones)
        self.vit = ViTTokenBranch(ckpt_path=vit_ckpt, freeze=freeze_backbones, drop_path_rate=drop_path_rate)

        self.fusion = TokenCrossFusion(
            cnn_dim=self.cnn.feature_dim,
            vit_dim=self.vit.feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            num_classes=num_classes
        )

        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n[DBFN-TOKENS] Params:")
        print(f"  Total:       {total/1e6:.1f}M")
        print(f"  Trainables:  {trainable/1e6:.1f}M")

    def freeze_all_backbones(self):
        self.cnn.freeze_all()
        self.vit.freeze_all()

    def unfreeze_partial(self, resnet_stages: int = 1, vit_blocks: int = 2):
        self.cnn.unfreeze_last_layers(num_stages=resnet_stages)
        self.vit.unfreeze_last_blocks(num_blocks=vit_blocks)

    def unfreeze_all(self):
        self.cnn.unfreeze_all()
        self.vit.unfreeze_all()

    def get_optimizer_groups(self, lr_backbone: float, lr_fusion: float, weight_decay: float):
        backbone_params = []
        fusion_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("fusion."):
                fusion_params.append(p)
            else:
                backbone_params.append(p)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
        groups.append({"params": fusion_params, "lr": lr_fusion, "weight_decay": weight_decay})
        return groups

    def forward(self, x):
        cnn_tokens = self.cnn(x)
        vit_tokens = self.vit(x)
        return self.fusion(cnn_tokens, vit_tokens)


# ============================================================
# DATA
# ============================================================

def make_weighted_sampler_from_subset(full_train: FER2013Dataset, indices: np.ndarray):
    labels = np.array([full_train.samples[i][1] for i in indices])
    class_counts = np.bincount(labels, minlength=len(FER2013_CLASSES)).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    print("[SAMPLER] WeightedRandomSampler activé")
    for i, name in enumerate(FER2013_CLASSES):
        print(f"  {name:<10} count={int(class_counts[i]):5d} weight={class_weights[i]:.4f}")
    return sampler


def create_dataloaders(args):
    print("[DATA] Chargement dataset neutre (transform=None) pour split...")
    full_train = FER2013Dataset(root_dir=args.data_dir, split="train", transform=None)

    y = np.array([lab for _, lab in full_train.samples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))

    print("[DATA] Split stratifié:")
    print(f"  - Train: {len(train_idx)} images ({100*(1-args.val_ratio):.0f}%)")
    print(f"  - Val:   {len(val_idx)} images ({100*args.val_ratio:.0f}%)")

    train_tf = get_train_transforms(img_size=224, use_augmentation=True)
    val_tf = get_val_transforms(img_size=224)

    train_ds_full = FER2013Dataset(root_dir=args.data_dir, split="train", transform=train_tf)
    val_ds_full   = FER2013Dataset(root_dir=args.data_dir, split="train", transform=val_tf)
    test_ds       = FER2013Dataset(root_dir=args.data_dir, split="test",  transform=val_tf)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds   = Subset(val_ds_full,   val_idx)

    print(f"  - Test:  {len(test_ds)} images (évaluation finale)")

    sampler = None
    shuffle = True
    if args.weighted_sampler:
        sampler = make_weighted_sampler_from_subset(full_train, train_idx)
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


# ============================================================
# MIXUP
# ============================================================

def mixup_data(x, y, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_mixup=False, mixup_alpha=0.1):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            # métriques train sur mixup = bruyantes; ici on ne les calcule pas strictement
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
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(1, len(loader))

    if (not use_mixup) and len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = 100.0 * np.mean(all_preds == all_labels)
        f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    else:
        acc, f1 = 0.0, 0.0

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

    val_loss = running_loss / max(1, len(loader))
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_acc = 100.0 * np.mean(all_preds == all_labels)
    val_f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, val_f1, all_preds, all_labels


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Test", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = 100.0 * np.mean(all_preds == all_labels)
    test_f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return test_acc, test_f1, all_preds, all_labels


def save_checkpoint(model, optimizer, scheduler, epoch, val_f1, path, is_best=False):
    ensure_dir(os.path.dirname(path))
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_f1": float(val_f1),
    }
    torch.save(ckpt, path)
    if is_best:
        best_path = path.replace(".pth", "_best.pth")
        torch.save(ckpt, best_path)


# ============================================================
# MAIN
# ============================================================

def build_optimizer_and_scheduler(args, model, only_fusion: bool):
    """
    - only_fusion=True: n'optimise que les params trainables (souvent fusion only quand backbones gelés)
    - only_fusion=False: utilise groupes backbone/fusion (LR diff)
    """
    if only_fusion:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=args.lr_fusion, weight_decay=args.weight_decay)
        print(f"[OPTIM] AdamW (fusion only) lr={args.lr_fusion:.2e}")
    else:
        groups = model.get_optimizer_groups(args.lr_backbone, args.lr_fusion, args.weight_decay)
        optimizer = optim.AdamW(groups)
        print(f"[OPTIM] AdamW (groups) lr_backbone={args.lr_backbone:.2e} lr_fusion={args.lr_fusion:.2e}")

    warmup_epochs = min(args.warmup_epochs, max(1, args.epochs // 5))
    warm = LinearLR(optimizer, start_factor=args.warmup_start_factor, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_epochs])

    return optimizer, scheduler


def main(args):
    print("\n" + "=" * 70)
    print("🔀 ENTRAÎNEMENT DBFN TOKENS (ResNet50 tokens + ViT tokens) - FER2013")
    print("=" * 70)

    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.results_dir)
    ensure_dir(args.log_dir)

    set_seed(args.seed)
    device = get_device(args.device)

    freeze_mode = "GELÉS" if args.freeze_backbones else "FINE-TUNING"
    print("\n[CONFIG]")
    print(f"  Backbones: {freeze_mode}")
    print(f"  CNN ckpt: {args.cnn_checkpoint}")
    print(f"  ViT ckpt: {args.vit_checkpoint}")
    print(f"  epochs={args.epochs} bs={args.batch_size}")
    print(f"  lr_backbone={args.lr_backbone} lr_fusion={args.lr_fusion}")
    print(f"  d_model={args.d_model} heads={args.num_heads} dropout={args.dropout}")
    print(f"  label_smoothing={args.label_smoothing} wd={args.weight_decay}")
    print(f"  mixup={args.mixup} weighted_sampler={args.weighted_sampler}")
    print(f"  val_ratio={args.val_ratio}")
    print(f"  unfreeze_epoch={args.unfreeze_epoch} resnet_layers={args.unfreeze_resnet_layers} vit_blocks={args.unfreeze_vit_blocks}")

    # DATA
    train_loader, val_loader, test_loader = create_dataloaders(args)

    # MODEL
    print("\n[MODEL] Création DBFN Tokens...")
    model = DBFNTokens(
        cnn_ckpt=args.cnn_checkpoint,
        vit_ckpt=args.vit_checkpoint,
        freeze_backbones=args.freeze_backbones,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        num_classes=7
    ).to(device)

    # LOSS
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # OPTIM + SCHED
    # Si freeze_backbones=True, on optimise fusion only au départ.
    # Si no_freeze, on optimise groupes dès le début.
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, only_fusion=args.freeze_backbones)

    # TensorBoard
    exp_name = f"dbfn_tokens_{'frozen' if args.freeze_backbones else 'finetune'}_{int(time.time())}"
    writer = SummaryWriter(os.path.join(args.log_dir, exp_name))

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # MIXUP: on évite mixup si weighted_sampler (sinon ça fait un combo pas stable)
    use_mixup = (args.mixup > 0) and (not args.weighted_sampler)

    print("\n" + "=" * 70)
    print("🚀 DÉBUT ENTRAÎNEMENT")
    print("=" * 70 + "\n")

    start_time = time.time()

    for epoch in range(args.epochs):
        ep_t0 = time.time()

        # OPTION 2: unfreeze progressif au bon moment (si on est parti frozen)
        if args.freeze_backbones and (args.unfreeze_epoch > 0) and ((epoch + 1) == args.unfreeze_epoch):
            print("\n[UNFREEZE] Début Phase B: unfreeze partiel backbones + rebuild optimizer/scheduler")
            model.unfreeze_partial(
                resnet_stages=args.unfreeze_resnet_layers,
                vit_blocks=args.unfreeze_vit_blocks
            )
            optimizer, scheduler = build_optimizer_and_scheduler(args, model, only_fusion=False)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mixup=use_mixup, mixup_alpha=args.mixup
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )

        # IMPORTANT: appeler scheduler.step() sans epoch (évite warning)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # LR affichage: on prend le dernier groupe (fusion) pour suivre
        current_lr = optimizer.param_groups[-1]["lr"]

        # Gap si train_acc dispo (si mixup actif train_acc=0)
        gap = (train_acc - val_acc) if train_acc > 0 else 0.0
        ep_time = time.time() - ep_t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} ({ep_time:.1f}s) | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% F1 {train_f1:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% F1 {val_f1:.2f}% | "
            f"LR {current_lr:.2e}" + (f" | Gap {gap:.2f}%" if gap > 0 else "")
        )

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        if train_acc > 0:
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("F1/train", train_f1, epoch)
            writer.add_scalar("OverfitGap/acc", gap, epoch)
        writer.add_scalar("LR/fusion_group", current_lr, epoch)

        # Checkpoint
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            patience_counter = 0
            print(f"  ✓ Nouveau meilleur modèle! (Val F1: {val_f1:.2f}%)")
        else:
            patience_counter += 1

        ckpt_path = os.path.join(args.checkpoint_dir, f"{exp_name}_epoch{epoch+1}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, val_f1, ckpt_path, is_best=is_best)

        # Early stop
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n[EARLY STOP] Pas d'amélioration depuis {args.patience} epochs.")
            break

    total_time = time.time() - start_time

    # Load best
    print("\n" + "=" * 70)
    print("📊 ÉVALUATION FINALE SUR TEST SET")
    print("=" * 70)

    best_path = os.path.join(args.checkpoint_dir, f"{exp_name}_epoch{best_epoch+1}_best.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Best loaded: epoch {best_epoch+1}, Val F1: {best_val_f1:.2f}%")
    else:
        print("[WARN] Best checkpoint introuvable -> évaluation sur dernier état")

    test_acc, test_f1, test_preds, test_labels = evaluate_test(model, test_loader, device)

    print("\n🎯 TEST RESULTS:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Macro-F1:  {test_f1:.2f}%")

    print_classification_report(test_labels, test_preds, "DBFN_TOKENS")

    # Figures
    cm_path = os.path.join(args.results_dir, f"{exp_name}_confusion_matrix.png")
    acc_path = os.path.join(args.results_dir, f"{exp_name}_per_class_accuracy.png")
    curves_path = os.path.join(args.results_dir, f"{exp_name}_training_curves.png")

    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=cm_path,
        title="Matrice de Confusion - DBFN_TOKENS (Test)"
    )
    plot_per_class_accuracy(
        test_labels, test_preds,
        save_path=acc_path,
        title="Accuracy par Classe - DBFN_TOKENS (Test)"
    )
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=curves_path
    )

    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"  Exp: {exp_name}")
    print(f"  Best Val F1: {best_val_f1:.2f}% (epoch {best_epoch+1})")
    print(f"  Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.2f}%")
    print(f"  Temps total: {total_time/60:.1f} min")
    print("=" * 70)

    writer.close()
    return 0


# ============================================================
# ARGS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train DBFN TOKENS (ResNet50 tokens + ViT tokens) sur FER2013",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    p.add_argument("--data_dir", type=str, default="data/fer2013")
    p.add_argument("--cnn_checkpoint", type=str, required=True)
    p.add_argument("--vit_checkpoint", type=str, required=True)

    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_hybrid_tokens")
    p.add_argument("--results_dir", type=str, default="results_hybrid_tokens")
    p.add_argument("--log_dir", type=str, default="logs_hybrid_tokens")

    # Freeze strategy
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze_backbones", action="store_true")
    g.add_argument("--no_freeze", action="store_true")

    # Tokens fusion params
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--drop_path_rate", type=float, default=0.1)

    # Train
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_backbone", type=float, default=2e-6)
    p.add_argument("--lr_fusion", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--mixup", type=float, default=0.0)

    # Split / Loader
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--weighted_sampler", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)

    # Scheduler
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--warmup_start_factor", type=float, default=0.1)
    p.add_argument("--min_lr", type=float, default=1e-7)

    # Early stopping
    p.add_argument("--patience", type=int, default=12)

    # Option 2: progressive unfreeze (utile surtout si start frozen)
    p.add_argument("--unfreeze_epoch", type=int, default=0,
                   help="Epoch (1-based) où on passe en Phase B: unfreeze partiel + rebuild optimizer")
    p.add_argument("--unfreeze_resnet_layers", type=int, default=1,
                   help="Nb de stages ResNet à dégeler depuis la fin (1=layer4, 2=layer3+4, ...)")
    p.add_argument("--unfreeze_vit_blocks", type=int, default=2,
                   help="Nb de blocks ViT à dégeler depuis la fin")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])

    args = p.parse_args()
    args.freeze_backbones = bool(args.freeze_backbones)
    return args


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.data_dir):
        print(f"❌ Dataset non trouvé: {args.data_dir}")
        sys.exit(1)
    if not os.path.exists(args.cnn_checkpoint):
        print(f"❌ CNN checkpoint introuvable: {args.cnn_checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.vit_checkpoint):
        print(f"❌ ViT checkpoint introuvable: {args.vit_checkpoint}")
        sys.exit(1)

    sys.exit(main(args))