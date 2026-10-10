"""
Architectures ResNet-18 et ResNet-50 pour FER2013
==================================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG

Basé sur: He et al. (2016) "Deep Residual Learning for Image Recognition"
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple


class ResNetFER(nn.Module):
    """
    ResNet adapté pour la reconnaissance des expressions faciales.
    
    Modifications par rapport au ResNet standard:
    1. Couche FC finale remplacée pour 7 classes (émotions)
    2. Dropout ajouté avant la classification
    3. Support pour le gel des couches (freeze backbone)
    
    Args:
        model_name: 'resnet18' ou 'resnet50'
        num_classes: Nombre de classes (7 pour FER2013)
        pretrained: Utiliser les poids pré-entraînés ImageNet
        dropout_rate: Taux de dropout avant la couche finale
        freeze_backbone: Geler les couches convolutives
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(ResNetFER, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Charger le modèle de base
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Modèle non supporté: {model_name}. Utilisez 'resnet18' ou 'resnet50'.")
        
        # Geler le backbone si demandé
        if freeze_backbone:
            self._freeze_backbone()
        
        # Remplacer la couche de classification
        self.backbone.fc = nn.Identity()  # Supprimer la couche FC originale
        
        # Nouvelle tête de classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Initialiser les poids de la nouvelle couche
        self._init_classifier()
    
    def _freeze_backbone(self):
        """Geler toutes les couches sauf la couche finale."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print(f"[INFO] Backbone gelé - seule la couche de classification sera entraînée")
    
    def _init_classifier(self):
        """Initialiser les poids du classifier avec Xavier/Glorot."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor d'entrée [batch_size, 3, 224, 224]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        # Extraction de features via le backbone
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extraire les features avant la classification.
        Utile pour la visualisation et l'analyse.
        
        Args:
            x: Tensor d'entrée
        
        Returns:
            Features [batch_size, feature_dim]
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def get_num_params(self) -> Tuple[int, int]:
        """
        Compter le nombre de paramètres.
        
        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def summary(self):
        """Afficher un résumé du modèle."""
        total, trainable = self.get_num_params()
        print("=" * 60)
        print(f"Modèle: {self.model_name}")
        print(f"Classes: {self.num_classes}")
        print(f"Pré-entraîné: {self.pretrained}")
        print(f"Dimension features: {self.feature_dim}")
        print("-" * 60)
        print(f"Paramètres totaux: {total:,}")
        print(f"Paramètres entraînables: {trainable:,}")
        print(f"Paramètres gelés: {total - trainable:,}")
        print("=" * 60)


def create_resnet18(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False
) -> ResNetFER:
    """
    Créer un modèle ResNet-18 pour FER.
    
    ResNet-18 Architecture:
    - 4 stages avec [2, 2, 2, 2] blocs résiduels
    - ~11.2M paramètres
    - Idéal pour: datasets de taille moyenne, entraînement rapide
    """
    return ResNetFER(
        model_name="resnet18",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )


def create_resnet50(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False
) -> ResNetFER:
    """
    Créer un modèle ResNet-50 pour FER.
    
    ResNet-50 Architecture:
    - 4 stages avec [3, 4, 6, 3] blocs bottleneck
    - ~23.5M paramètres
    - Idéal pour: meilleure accuracy, plus de capacité
    """
    return ResNetFER(
        model_name="resnet50",
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )


# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TEST DES MODÈLES RESNET POUR FER")
    print("=" * 60)
    
    # Tester ResNet-18
    print("\n[TEST] ResNet-18")
    model_18 = create_resnet18(pretrained=True)
    model_18.summary()
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model_18(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Vérifier les dimensions
    assert output.shape == (4, 7), "Erreur: dimension de sortie incorrecte"
    print("✓ Forward pass OK\n")
    
    # Tester ResNet-50
    print("[TEST] ResNet-50")
    model_50 = create_resnet50(pretrained=True)
    model_50.summary()
    
    output = model_50(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 7), "Erreur: dimension de sortie incorrecte"
    print("✓ Forward pass OK\n")
    
    # Test extraction de features
    print("[TEST] Extraction de features")
    features = model_18.get_features(x)
    print(f"Features shape (ResNet-18): {features.shape}")
    
    features = model_50.get_features(x)
    print(f"Features shape (ResNet-50): {features.shape}")
    print("✓ Feature extraction OK\n")
    
    print("=" * 60)
    print("TOUS LES TESTS PASSÉS ✓")
    print("=" * 60)
