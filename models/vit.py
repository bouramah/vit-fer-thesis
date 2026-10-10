"""
Vision Transformer (ViT) pour FER2013
======================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG

Basé sur: Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words"

Architecture ViT:
1. Patch Embedding: Découpe l'image en patches (16x16)
2. Position Embedding: Ajoute l'information de position
3. Transformer Encoder: Blocs de Self-Attention + MLP
4. Classification Head: MLP pour prédire les émotions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ==================== PATCH EMBEDDING ====================

class PatchEmbedding(nn.Module):
    """
    Découpe l'image en patches et les projette en embeddings.
    
    Image [B, 3, 224, 224] → Patches [B, num_patches, embed_dim]
    
    Avec patch_size=16 et img_size=224:
    - num_patches = (224/16)² = 14² = 196 patches
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196
        
        # Projection linéaire via Conv2d (équivalent à découper + projeter)
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            [B, 196, 768]
        """
        # [B, 3, 224, 224] → [B, 768, 14, 14]
        x = self.projection(x)
        
        # [B, 768, 14, 14] → [B, 768, 196] → [B, 196, 768]
        x = x.flatten(2).transpose(1, 2)
        
        return x


# ==================== MULTI-HEAD SELF-ATTENTION ====================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    
    Permet au modèle d'apprendre différentes relations
    entre les patches simultanément.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        # Projections Q, K, V combinées
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        # Projection de sortie
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] où N = num_patches + 1 (CLS token)
        Returns:
            output: [B, N, D]
            attention_weights: [B, num_heads, N, N]
        """
        B, N, D = x.shape
        
        # Calculer Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores: (Q @ K^T) / sqrt(d_k)
        attention = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        
        # Attention output: Attention @ V
        x = (attention @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        # Projection finale
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x, attention


# ==================== MLP (Feed-Forward Network) ====================

class MLP(nn.Module):
    """
    Feed-Forward Network dans le Transformer.
    
    Expand → GELU → Project back
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ==================== TRANSFORMER ENCODER BLOCK ====================

class TransformerEncoderBlock(nn.Module):
    """
    Un bloc du Transformer Encoder.
    
    LayerNorm → MHSA → Residual → LayerNorm → MLP → Residual
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-Attention avec résidu
        attn_out, attention_weights = self.attention(self.norm1(x))
        x = x + attn_out
        
        # MLP avec résidu
        x = x + self.mlp(self.norm2(x))
        
        return x, attention_weights


# ==================== VISION TRANSFORMER ====================

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) pour la classification d'images.
    
    Architecture:
    1. Patch Embedding
    2. [CLS] Token + Position Embedding
    3. Transformer Encoder (L blocks)
    4. Classification Head
    
    Variantes:
    - ViT-Tiny:  embed_dim=192,  depth=12, heads=3   (~5.7M params)
    - ViT-Small: embed_dim=384,  depth=12, heads=6   (~22M params)
    - ViT-Base:  embed_dim=768,  depth=12, heads=12  (~86M params)
    - ViT-Large: embed_dim=1024, depth=24, heads=16  (~307M params)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 7,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        classifier_dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS Token: token spécial pour la classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position Embedding: apprend la position de chaque patch
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Dropout après embedding
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
        # Layer Norm finale
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids (comme dans le papier original)."""
        # Position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass.
        
        Args:
            x: [B, 3, 224, 224]
            return_attention: Si True, retourne aussi les poids d'attention
        
        Returns:
            logits: [B, num_classes]
            attention_weights (optionnel): Liste de [B, num_heads, N, N]
        """
        B = x.shape[0]
        
        # Patch Embedding: [B, 3, 224, 224] → [B, 196, 768]
        x = self.patch_embed(x)
        
        # Ajouter le CLS token: [B, 196, 768] → [B, 197, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Ajouter Position Embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer Encoder
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            if return_attention:
                attention_weights.append(attn)
        
        # Layer Norm
        x = self.norm(x)
        
        # Classification: utiliser seulement le CLS token
        cls_output = x[:, 0]  # [B, 768]
        logits = self.classifier(cls_output)  # [B, 7]
        
        if return_attention:
            return logits, attention_weights
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtenir les cartes d'attention pour la visualisation.
        
        Returns:
            Attention du dernier bloc: [B, num_heads, N, N]
        """
        _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights[-1]
    
    def get_num_params(self) -> Tuple[int, int]:
        """Compter les paramètres."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def summary(self):
        """Afficher un résumé du modèle."""
        total, trainable = self.get_num_params()
        print("=" * 60)
        print("VISION TRANSFORMER (ViT)")
        print("=" * 60)
        print(f"  Image size: 224x224")
        print(f"  Patch size: {self.patch_embed.patch_size}x{self.patch_embed.patch_size}")
        print(f"  Num patches: {self.patch_embed.num_patches}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Depth: {self.depth} blocks")
        print(f"  Num heads: {self.num_heads}")
        print(f"  Num classes: {self.num_classes}")
        print("-" * 60)
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print("=" * 60)


# ==================== FACTORY FUNCTIONS ====================

def create_vit_tiny(num_classes: int = 7, dropout: float = 0.1, **kwargs) -> VisionTransformer:
    """
    ViT-Tiny: Plus petit modèle, rapide à entraîner.
    ~5.7M paramètres
    """
    return VisionTransformer(
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=dropout,
        **kwargs
    )


def create_vit_small(num_classes: int = 7, dropout: float = 0.1, **kwargs) -> VisionTransformer:
    """
    ViT-Small: Bon compromis taille/performance.
    ~22M paramètres
    """
    return VisionTransformer(
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=dropout,
        **kwargs
    )


def create_vit_base(num_classes: int = 7, dropout: float = 0.1, **kwargs) -> VisionTransformer:
    """
    ViT-Base: Modèle standard du papier original.
    ~86M paramètres
    """
    return VisionTransformer(
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=dropout,
        **kwargs
    )


# ==================== VIT AVEC POIDS PRÉ-ENTRAÎNÉS ====================

def create_vit_pretrained(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 7,
    dropout: float = 0.5,
    pretrained: bool = True
) -> nn.Module:
    """
    Créer un ViT avec poids pré-entraînés ImageNet via timm.
    
    Args:
        model_name: Nom du modèle timm
            - 'vit_tiny_patch16_224'
            - 'vit_small_patch16_224'
            - 'vit_base_patch16_224'
            - 'deit_small_patch16_224' (Data-efficient ViT)
            - 'deit_base_patch16_224'
        num_classes: Nombre de classes (7 pour FER2013)
        dropout: Dropout avant classification
        pretrained: Utiliser les poids ImageNet
    
    Returns:
        Modèle ViT prêt pour le fine-tuning
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm requis: pip install timm")
    
    # Créer le modèle avec poids pré-entraînés
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    
    return model


class ViTForFER(nn.Module):
    """
    Wrapper pour ViT pré-entraîné adapté à FER2013.
    
    Utilise timm pour charger les poids ImageNet.
    
    Améliorations v2:
    - drop_path_rate: Stochastic Depth pour réduire l'overfitting
    - drop_rate: Dropout standard
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.5,
        drop_path_rate: float = 0.1,  # 🆕 Stochastic Depth
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm requis: pip install timm")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.drop_path_rate = drop_path_rate
        
        # Charger le modèle pré-entraîné avec drop_path
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Enlever la tête de classification
            drop_rate=dropout,
            drop_path_rate=drop_path_rate  # 🆕 Stochastic Depth (très efficace!)
        )
        
        # Dimension de l'embedding
        self.embed_dim = self.backbone.num_features
        
        # Nouvelle tête de classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        # Geler le backbone si demandé
        if freeze_backbone:
            self._freeze_backbone()
        
        # Initialiser le classifier
        self._init_classifier()
    
    def _freeze_backbone(self):
        """Geler le backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[INFO] Backbone ViT gelé")
    
    def _init_classifier(self):
        """Initialiser le classifier."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extraire les features
        features = self.backbone(x)  # [B, embed_dim]
        
        # Classification
        logits = self.classifier(features)  # [B, num_classes]
        
        return logits
    
    def get_num_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def summary(self):
        total, trainable = self.get_num_params()
        print("=" * 60)
        print(f"ViT FOR FER - {self.model_name}")
        print("=" * 60)
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Drop path rate: {self.drop_path_rate}")
        print("-" * 60)
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print("=" * 60)


# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TEST DES MODÈLES VISION TRANSFORMER")
    print("=" * 60)
    
    # Test input
    x = torch.randn(4, 3, 224, 224)
    
    # Test ViT-Tiny (from scratch)
    print("\n[TEST] ViT-Tiny (from scratch)")
    model = create_vit_tiny(num_classes=7)
    model.summary()
    output = model(x)
    print(f"Input: {x.shape} → Output: {output.shape}")
    assert output.shape == (4, 7), "Erreur de dimension"
    print("✓ OK\n")
    
    # Test ViT-Small
    print("[TEST] ViT-Small (from scratch)")
    model = create_vit_small(num_classes=7)
    model.summary()
    output = model(x)
    print(f"Input: {x.shape} → Output: {output.shape}")
    assert output.shape == (4, 7), "Erreur de dimension"
    print("✓ OK\n")
    
    # Test ViT-Base
    print("[TEST] ViT-Base (from scratch)")
    model = create_vit_base(num_classes=7)
    model.summary()
    output = model(x)
    print(f"Input: {x.shape} → Output: {output.shape}")
    assert output.shape == (4, 7), "Erreur de dimension"
    print("✓ OK\n")
    
    # Test attention maps
    print("[TEST] Extraction des attention maps")
    logits, attention = model(x, return_attention=True)
    print(f"Nombre de blocs: {len(attention)}")
    print(f"Shape attention (dernier bloc): {attention[-1].shape}")
    print("✓ OK\n")
    
    # Test ViT pré-entraîné (si timm installé)
    try:
        print("[TEST] ViT pré-entraîné (timm)")
        model = ViTForFER(
            model_name="vit_small_patch16_224",
            num_classes=7,
            pretrained=True
        )
        model.summary()
        output = model(x)
        print(f"Input: {x.shape} → Output: {output.shape}")
        print("✓ OK\n")
    except ImportError:
        print("⚠️ timm non installé, test ignoré\n")
    
    print("=" * 60)
    print("TOUS LES TESTS PASSÉS ✓")
    print("=" * 60)
