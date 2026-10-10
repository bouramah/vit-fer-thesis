#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual-Branch Fusion Network (DBFN) pour FER2013
==============================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG

Architecture hybride combinant:
- Branche CNN: ResNet-50 (features locales)
- Branche ViT: ViT-Base (features globales)
- Module de fusion: Concat+MLP (baseline) ou Cross-Attention (expérimental)

Notes importantes:
- Chargement de checkpoints robuste (gère model_state_dict, module., backbone., classifier/head/fc)
- ViT drop_path_rate forcé à 0.0 si backbones gelés (features stables)
- Cross-attention sur 1 token par branche est peu expressive; concat est le meilleur baseline
"""

import torch
import torch.nn as nn
import timm
from torchvision import models


# ============================================================
# UTILS : state_dict loading helpers
# ============================================================

def _extract_state_dict(ckpt_obj):
    """Supporte ckpt dict ou state_dict direct."""
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def _strip_prefix(key: str, prefix: str) -> str:
    if key.startswith(prefix):
        return key[len(prefix):]
    return key


def _sanitize_key(k: str) -> str:
    """Enlève prefix module. si DataParallel."""
    if k.startswith("module."):
        k = k.replace("module.", "", 1)
    return k


# ============================================================
# BRANCHE CNN : ResNet-50
# ============================================================

class ResNet50Branch(nn.Module):
    """
    Branche CNN utilisant torchvision ResNet-50.
    Extrait des features de dimension 2048 (après global avg pool) via fc=Identity().
    """

    def __init__(self, pretrained_checkpoint=None, freeze=False):
        super().__init__()

        # ResNet-50 sans poids ImageNet par défaut (on charge NOS poids si checkpoint fourni)
        self.resnet = models.resnet50(weights=None)
        self.feature_dim = self.resnet.fc.in_features  # 2048

        # Supprimer la couche FC (on veut juste les features)
        self.resnet.fc = nn.Identity()

        if pretrained_checkpoint:
            self._load_fer_checkpoint(pretrained_checkpoint)

        if freeze:
            self.freeze()

    def _load_fer_checkpoint(self, checkpoint_path: str):
        """Charge les poids ResNet depuis un checkpoint de TON pipeline."""
        print(f"[CNN Branch] Chargement checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(ckpt)

        filtered = {}
        for k, v in state_dict.items():
            k = _sanitize_key(k)

            # certains wrappers mettent backbone.
            if k.startswith("backbone."):
                k = _strip_prefix(k, "backbone.")

            # ignorer toute tête de classification
            if k.startswith("fc.") or k.startswith("head.") or ("classifier" in k):
                continue

            filtered[k] = v

        missing, unexpected = self.resnet.load_state_dict(filtered, strict=False)
        print(f"  Chargé! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if unexpected:
            print(f"  Unexpected keys (ex 5): {unexpected[:5]}")
        if missing:
            # fc.* sera généralement "missing" car on a remplacé par Identity (normal)
            # mais si on a trop de missing, c'est suspect
            if len(missing) > 50:
                print(f"  ⚠️ Beaucoup de missing keys (ex 5): {missing[:5]}")

    def freeze(self):
        for p in self.resnet.parameters():
            p.requires_grad = False
        print("[CNN Branch] Backbone gelé")

    def unfreeze(self):
        for p in self.resnet.parameters():
            p.requires_grad = True
        print("[CNN Branch] Backbone dégelé")

    def forward(self, x):
        # x: [B, 3, 224, 224] -> features: [B, 2048]
        return self.resnet(x)


# ============================================================
# BRANCHE ViT : ViT-Base
# ============================================================

class ViTBaseBranch(nn.Module):
    """
    Branche Transformer ViT-Base/16 (timm).
    Retourne un embedding global de dimension 768.
    """

    def __init__(self, pretrained_checkpoint=None, freeze=False, drop_path_rate=0.1,
                 imagenet_fallback=True):
        super().__init__()

        # Si on gèle les backbones, on veut un embedding stable -> drop_path = 0
        if freeze:
            drop_path_rate = 0.0

        # Modèle sans head de classification: num_classes=0 -> retourne features
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate
        )
        self.feature_dim = 768

        if pretrained_checkpoint:
            self._load_fer_checkpoint(pretrained_checkpoint)
        elif imagenet_fallback:
            # fallback ImageNet si aucun checkpoint fourni (nécessite internet)
            print("[ViT Branch] Aucun checkpoint FER fourni, chargement ImageNet...")
            vit_imagenet = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=0,
                drop_path_rate=drop_path_rate
            )
            self.vit.load_state_dict(vit_imagenet.state_dict(), strict=True)
            del vit_imagenet
        else:
            # déploiement hors-ligne : init aléatoire, écrasée par le checkpoint DBFN complet
            print("[ViT Branch] Init sans ImageNet (imagenet_fallback=False) : poids fournis par le checkpoint DBFN.")

        if freeze:
            self.freeze()

    def _load_fer_checkpoint(self, checkpoint_path: str):
        """Charge les poids ViT depuis ton checkpoint ViTForFER."""
        print(f"[ViT Branch] Chargement checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(ckpt)

        filtered = {}
        for k, v in state_dict.items():
            k = _sanitize_key(k)

            # ton wrapper ViTForFER utilise souvent backbone./classifier.
            if k.startswith("backbone."):
                k = _strip_prefix(k, "backbone.")
            elif k.startswith("encoder."):
                # si jamais un autre wrapper
                k = _strip_prefix(k, "encoder.")

            # ignorer têtes de classification
            if k.startswith("classifier.") or k.startswith("head.") or ("classifier" in k) or ("head" in k):
                continue

            # ignorer fc.* si jamais présent
            if k.startswith("fc."):
                continue

            filtered[k] = v

        if len(filtered) == 0:
            print("  ⚠️ Filtrage vide: tentative de chargement direct (sans head/classifier)")
            filtered = {
                _sanitize_key(k): v
                for k, v in state_dict.items()
                if ("classifier" not in k and "head" not in k and not k.startswith("fc."))
            }

        missing, unexpected = self.vit.load_state_dict(filtered, strict=False)
        print(f"  Chargé! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if unexpected:
            print(f"  Unexpected keys (ex 5): {unexpected[:5]}")
        if missing and len(missing) > 50:
            print(f"  ⚠️ Beaucoup de missing keys (ex 5): {missing[:5]}")

    def freeze(self):
        for p in self.vit.parameters():
            p.requires_grad = False
        print("[ViT Branch] Backbone gelé")

    def unfreeze(self):
        for p in self.vit.parameters():
            p.requires_grad = True
        print("[ViT Branch] Backbone dégelé")

    def forward(self, x):
        # x: [B, 3, 224, 224] -> features: [B, 768]
        return self.vit(x)


# ============================================================
# MODULE DE FUSION : Cross-Attention (expérimental)
# ============================================================

class CrossAttentionFusion(nn.Module):
    """
    Fusion par cross-attention bidirectionnelle sur des embeddings globaux.
    ATTENTION: ici seq_len=1, donc l'attention est peu expressive.
    Conserver surtout pour tests; concat est le baseline recommandé.
    """

    def __init__(
        self,
        cnn_dim=2048,
        vit_dim=768,
        proj_dim=512,
        num_heads=8,
        dropout=0.3,
        num_classes=7
    ):
        super().__init__()

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.cross_attn_cnn2vit = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_cnn2vit = nn.LayerNorm(proj_dim)

        self.cross_attn_vit2cnn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_vit2cnn = nn.LayerNorm(proj_dim)

        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim // 2, num_classes)
        )

    def forward(self, cnn_feat, vit_feat):
        cnn_proj = self.cnn_proj(cnn_feat)  # [B, proj_dim]
        vit_proj = self.vit_proj(vit_feat)  # [B, proj_dim]

        # seq_len=1
        cnn_seq = cnn_proj.unsqueeze(1)  # [B, 1, proj_dim]
        vit_seq = vit_proj.unsqueeze(1)  # [B, 1, proj_dim]

        cnn_attended, _ = self.cross_attn_cnn2vit(query=cnn_seq, key=vit_seq, value=vit_seq)
        cnn_fused = self.norm_cnn2vit(cnn_seq + cnn_attended).squeeze(1)  # [B, proj_dim]

        vit_attended, _ = self.cross_attn_vit2cnn(query=vit_seq, key=cnn_seq, value=cnn_seq)
        vit_fused = self.norm_vit2cnn(vit_seq + vit_attended).squeeze(1)  # [B, proj_dim]

        gate_input = torch.cat([cnn_fused, vit_fused], dim=-1)  # [B, 2*proj_dim]
        gate_weight = self.gate(gate_input)  # [B, proj_dim]

        fused = torch.cat([gate_weight * cnn_fused, (1 - gate_weight) * vit_fused], dim=-1)
        return self.classifier(fused)


# ============================================================
# MODULE DE FUSION : Concat + MLP (baseline recommandé)
# ============================================================

class ConcatFusion(nn.Module):
    """
    Fusion simple: concaténation des features (2048 + 768) puis MLP.
    Baseline recommandé, stable et efficace.
    """

    def __init__(
        self,
        cnn_dim=2048,
        vit_dim=768,
        hidden_dim=512,
        dropout=0.3,
        num_classes=7
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, cnn_feat, vit_feat):
        combined = torch.cat([cnn_feat, vit_feat], dim=-1)  # [B, 2816]
        return self.classifier(combined)


# ============================================================
# MODULE DE FUSION : Gated Fusion (alternative légère)
# ============================================================

class GatedFusion(nn.Module):
    """
    Fusion par gating: les deux branches sont projetées dans un espace commun,
    puis combinées par une porte apprise (gate) élément par élément:
        fused = g * cnn_proj + (1 - g) * vit_proj
    Compromis entre concat (pas d'interaction) et cross-attention (lourd).
    Demandé explicitement par les reviewers (#15, R4.4).
    """

    def __init__(
        self,
        cnn_dim=2048,
        vit_dim=768,
        proj_dim=512,
        dropout=0.3,
        num_classes=7
    ):
        super().__init__()

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Porte élément par élément à partir des deux projections
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_classes),
        )

    def forward(self, cnn_feat, vit_feat):
        cnn_proj = self.cnn_proj(cnn_feat)  # [B, proj_dim]
        vit_proj = self.vit_proj(vit_feat)  # [B, proj_dim]
        g = self.gate(torch.cat([cnn_proj, vit_proj], dim=-1))  # [B, proj_dim]
        fused = g * cnn_proj + (1.0 - g) * vit_proj  # [B, proj_dim]
        return self.classifier(fused)


# ============================================================
# MODÈLE COMPLET : Dual-Branch Fusion Network
# ============================================================

class DualBranchFusionNetwork(nn.Module):
    """
    Combine ResNet-50 (CNN) + ViT-Base (Transformer) et une fusion configurable.

    Args:
        cnn_checkpoint: chemin vers checkpoint ResNet-50 (TON pipeline)
        vit_checkpoint: chemin vers checkpoint ViT-Base (TON pipeline)
        fusion_type: 'concat' (recommandé) ou 'cross_attention'
        freeze_backbones: True -> on entraîne seulement fusion
        proj_dim: hidden/proj dim pour fusion
        dropout: dropout fusion
        drop_path_rate: drop path pour ViT (auto=0 si freeze)
        num_classes: 7
    """

    def __init__(
        self,
        cnn_checkpoint=None,
        vit_checkpoint=None,
        fusion_type="concat",
        freeze_backbones=False,
        proj_dim=512,
        dropout=0.3,
        drop_path_rate=0.1,
        num_classes=7,
        imagenet_fallback=True
    ):
        super().__init__()

        self.fusion_type = fusion_type

        self.cnn_branch = ResNet50Branch(
            pretrained_checkpoint=cnn_checkpoint,
            freeze=freeze_backbones
        )

        self.vit_branch = ViTBaseBranch(
            pretrained_checkpoint=vit_checkpoint,
            freeze=freeze_backbones,
            drop_path_rate=drop_path_rate,
            imagenet_fallback=imagenet_fallback
        )

        cnn_dim = self.cnn_branch.feature_dim  # 2048
        vit_dim = self.vit_branch.feature_dim  # 768

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                cnn_dim=cnn_dim,
                vit_dim=vit_dim,
                proj_dim=proj_dim,
                num_heads=8,
                dropout=dropout,
                num_classes=num_classes
            )
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(
                cnn_dim=cnn_dim,
                vit_dim=vit_dim,
                hidden_dim=proj_dim,
                dropout=dropout,
                num_classes=num_classes
            )
        elif fusion_type == "gating":
            self.fusion = GatedFusion(
                cnn_dim=cnn_dim,
                vit_dim=vit_dim,
                proj_dim=proj_dim,
                dropout=dropout,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Fusion type non supporté: {fusion_type}")

        self._print_params()

    def _print_params(self):
        cnn_params = sum(p.numel() for p in self.cnn_branch.parameters())
        vit_params = sum(p.numel() for p in self.vit_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        total = cnn_params + vit_params + fusion_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n[DBFN] Architecture:")
        print(f"  CNN (ResNet-50):        {cnn_params/1e6:.1f}M params")
        print(f"  ViT (ViT-Base):         {vit_params/1e6:.1f}M params")
        print(f"  Fusion ({self.fusion_type}):    {fusion_params/1e6:.2f}M params")
        print(f"  Total:                  {total/1e6:.1f}M params")
        print(f"  Entraînables:           {trainable/1e6:.1f}M params")

    def freeze_backbones(self):
        self.cnn_branch.freeze()
        self.vit_branch.freeze()

    def unfreeze_backbones(self):
        self.cnn_branch.unfreeze()
        self.vit_branch.unfreeze()

    def get_optimizer_groups(self, lr_backbone=1e-5, lr_fusion=1e-4, weight_decay=0.05):
        backbone_params = []
        fusion_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fusion."):
                fusion_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
        groups.append({"params": fusion_params, "lr": lr_fusion, "weight_decay": weight_decay})
        return groups

    def forward(self, x):
        cnn_feat = self.cnn_branch(x)  # [B, 2048]
        vit_feat = self.vit_branch(x)  # [B, 768]
        logits = self.fusion(cnn_feat, vit_feat)  # [B, 7]
        return logits


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_dbfn(
    cnn_checkpoint=None,
    vit_checkpoint=None,
    fusion_type="concat",
    freeze_backbones=False,
    proj_dim=512,
    dropout=0.3,
    drop_path_rate=0.1,
    num_classes=7,
    imagenet_fallback=True
):
    return DualBranchFusionNetwork(
        cnn_checkpoint=cnn_checkpoint,
        vit_checkpoint=vit_checkpoint,
        fusion_type=fusion_type,
        freeze_backbones=freeze_backbones,
        proj_dim=proj_dim,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        num_classes=num_classes,
        imagenet_fallback=imagenet_fallback
    )