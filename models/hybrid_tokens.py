#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBFN Token Fusion (CNN tokens + ViT tokens) pour FER2013
========================================================
- CNN: ResNet-50 -> tokens spatiaux 7x7 = 49 tokens (2048D)
- ViT: ViT-Base -> tokens patch (cls + 196 patches) = 197 tokens (768D)
- Fusion: Cross-Attention bidirectionnelle entre séquences
- Pooling: mean pooling sur tokens (ou CLS pour ViT)
- Head: MLP -> 7 classes

Compatible checkpoints:
- ResNet50 de ton pipeline (avec fc/classifier) -> on charge backbone seulement
- ViTForFER (keys backbone.*) -> on charge backbone timm seulement
"""

import torch
import torch.nn as nn
import timm
from torchvision import models


# -----------------------------
# Utils state_dict
# -----------------------------
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _sanitize_key(k: str) -> str:
    if k.startswith("module."):
        k = k.replace("module.", "", 1)
    return k


# ============================================================
# CNN Branch (ResNet50) -> tokens [B, 49, 2048]
# ============================================================

class ResNet50TokenBranch(nn.Module):
    def __init__(self, pretrained_checkpoint=None, freeze=False):
        super().__init__()

        resnet = models.resnet50(weights=None)

        # On garde tout sauf avgpool/fc pour sortir une feature map [B, 2048, 7, 7]
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.feature_dim = 2048  # channels de layer4

        if pretrained_checkpoint:
            self._load_fer_checkpoint(pretrained_checkpoint)

        if freeze:
            self.freeze()

    def _load_fer_checkpoint(self, path: str):
        print(f"[CNN Tokens] Chargement checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(ckpt)

        filtered = {}
        for k, v in sd.items():
            k = _sanitize_key(k)

            # si wrapper backbone.
            if k.startswith("backbone."):
                k = k.replace("backbone.", "", 1)

            # ignorer head
            if k.startswith("fc.") or k.startswith("head.") or ("classifier" in k):
                continue

            filtered[k] = v

        # On doit charger dans un "resnet complet" ? Ici on a des sous-modules.
        # Astuce: reconstruire un resnet50 temporaire, charger, puis recopier les poids.
        tmp = models.resnet50(weights=None)
        missing, unexpected = tmp.load_state_dict(filtered, strict=False)
        print(f"  Chargé tmp ResNet! Missing={len(missing)} Unexpected={len(unexpected)}")

        # Copie vers nos modules
        self.stem[0].load_state_dict(tmp.conv1.state_dict())
        self.stem[1].load_state_dict(tmp.bn1.state_dict())
        self.layer1.load_state_dict(tmp.layer1.state_dict())
        self.layer2.load_state_dict(tmp.layer2.state_dict())
        self.layer3.load_state_dict(tmp.layer3.state_dict())
        self.layer4.load_state_dict(tmp.layer4.state_dict())

        del tmp

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        print("[CNN Tokens] Backbone gelé")

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        print("[CNN Tokens] Backbone dégelé")

    def forward(self, x):
        # x: [B,3,224,224]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B,2048,7,7] (224 input)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B,49,2048]
        return tokens


# ============================================================
# ViT Branch -> tokens [B, 197, 768]
# ============================================================

class ViTBaseTokenBranch(nn.Module):
    def __init__(self, pretrained_checkpoint=None, freeze=False, drop_path_rate=0.0):
        super().__init__()

        # Pour des tokens stables, drop_path à 0 conseillé si freeze
        if freeze:
            drop_path_rate = 0.0

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.feature_dim = 768

        if pretrained_checkpoint:
            self._load_fer_checkpoint(pretrained_checkpoint)
        else:
            print("[ViT Tokens] Aucun ckpt FER fourni: chargement ImageNet")
            vit_im = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0, drop_path_rate=drop_path_rate)
            self.vit.load_state_dict(vit_im.state_dict(), strict=True)
            del vit_im

        if freeze:
            self.freeze()

    def _load_fer_checkpoint(self, path: str):
        print(f"[ViT Tokens] Chargement checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(ckpt)

        filtered = {}
        for k, v in sd.items():
            k = _sanitize_key(k)

            # ton ViTForFER a souvent backbone.*
            if k.startswith("backbone."):
                k = k.replace("backbone.", "", 1)

            # ignorer head/classifier
            if k.startswith("classifier.") or k.startswith("head.") or ("classifier" in k) or ("head" in k):
                continue

            filtered[k] = v

        missing, unexpected = self.vit.load_state_dict(filtered, strict=False)
        print(f"  Chargé! Missing={len(missing)} Unexpected={len(unexpected)}")
        if missing and len(missing) > 50:
            print(f"  ⚠️ Beaucoup de missing keys (ex 5): {missing[:5]}")

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        print("[ViT Tokens] Backbone gelé")

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        print("[ViT Tokens] Backbone dégelé")

    def forward(self, x):
        """
        Retourne des tokens après norm: [B, 197, 768]
        On re-implémente la forward_features pour récupérer les tokens.
        """
        m = self.vit
        # patch embedding -> [B, 196, 768]
        x = m.patch_embed(x)

        # ajout cls + pos embed
        if getattr(m, "cls_token", None) is not None:
            cls = m.cls_token.expand(x.shape[0], -1, -1)  # [B,1,768]
            x = torch.cat((cls, x), dim=1)               # [B,197,768]

        # pos embed
        if getattr(m, "pos_embed", None) is not None:
            x = x + m.pos_embed

        x = m.pos_drop(x)
        x = m.blocks(x)
        x = m.norm(x)  # [B,197,768]
        return x


# ============================================================
# Fusion Token-based : Cross-Attention bidirectionnelle
# ============================================================

class TokenCrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=2048, vit_dim=768, d_model=512, num_heads=8, dropout=0.2, num_classes=7):
        super().__init__()

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention: CNN tokens query ViT tokens
        self.attn_cnn2vit = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_cnn = nn.LayerNorm(d_model)

        # Attention: ViT tokens query CNN tokens
        self.attn_vit2cnn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_vit = nn.LayerNorm(d_model)

        # Pooling + head
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, cnn_tokens, vit_tokens):
        """
        cnn_tokens: [B, 49, 2048]
        vit_tokens: [B, 197, 768]
        """
        cnn = self.cnn_proj(cnn_tokens)  # [B,49,d]
        vit = self.vit_proj(vit_tokens)  # [B,197,d]

        # CNN attends ViT
        cnn_att, _ = self.attn_cnn2vit(query=cnn, key=vit, value=vit)
        cnn_fused = self.norm_cnn(cnn + cnn_att)  # residual

        # ViT attends CNN
        vit_att, _ = self.attn_vit2cnn(query=vit, key=cnn, value=cnn)
        vit_fused = self.norm_vit(vit + vit_att)

        # Pooling
        # CNN: mean pooling
        cnn_vec = cnn_fused.mean(dim=1)  # [B,d]
        # ViT: utiliser CLS token (index 0) si présent
        vit_vec = vit_fused[:, 0, :]     # [B,d]

        fused = torch.cat([cnn_vec, vit_vec], dim=-1)  # [B,2d]
        logits = self.head(fused)
        return logits


# ============================================================
# Modèle complet
# ============================================================

class DBFNTokenFusion(nn.Module):
    def __init__(
        self,
        cnn_checkpoint=None,
        vit_checkpoint=None,
        freeze_backbones=True,
        d_model=512,
        num_heads=8,
        dropout=0.2,
        drop_path_rate=0.0,
        num_classes=7,
    ):
        super().__init__()

        self.cnn = ResNet50TokenBranch(pretrained_checkpoint=cnn_checkpoint, freeze=freeze_backbones)
        self.vit = ViTBaseTokenBranch(pretrained_checkpoint=vit_checkpoint, freeze=freeze_backbones, drop_path_rate=drop_path_rate)

        self.fusion = TokenCrossAttentionFusion(
            cnn_dim=2048, vit_dim=768,
            d_model=d_model, num_heads=num_heads,
            dropout=dropout, num_classes=num_classes
        )

        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n[DBFN-TOKENS] Params:")
        print(f"  Total:       {total/1e6:.1f}M")
        print(f"  Trainables:  {trainable/1e6:.1f}M")

    def get_optimizer_groups(self, lr_backbone=5e-6, lr_fusion=5e-4, weight_decay=0.05):
        backbone_params = []
        fusion_params = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("fusion."):
                fusion_params.append(p)
            else:
                backbone_params.append(p)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
        groups.append({"params": fusion_params, "lr": lr_fusion, "weight_decay": weight_decay})
        return groups

    def forward(self, x):
        cnn_tokens = self.cnn(x)          # [B,49,2048]
        vit_tokens = self.vit(x)          # [B,197,768]
        logits = self.fusion(cnn_tokens, vit_tokens)
        return logits


def create_dbfn_tokens(
    cnn_checkpoint=None,
    vit_checkpoint=None,
    freeze_backbones=True,
    d_model=512,
    num_heads=8,
    dropout=0.2,
    drop_path_rate=0.0,
    num_classes=7,
):
    return DBFNTokenFusion(
        cnn_checkpoint=cnn_checkpoint,
        vit_checkpoint=vit_checkpoint,
        freeze_backbones=freeze_backbones,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        num_classes=num_classes,
    )