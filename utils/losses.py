"""
Fonctions de Loss personnalisées pour FER2013
==============================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG

Contient:
- FocalLoss: Pour les datasets déséquilibrés
- LabelSmoothingCrossEntropy: Alternative à nn.CrossEntropyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss pour les datasets déséquilibrés.
    
    Papier: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    La Focal Loss réduit la contribution des exemples "faciles" (bien classés)
    et se concentre sur les exemples "difficiles" (mal classés).
    
    Formule: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: Facteur de focalisation (défaut: 2.0)
               - γ = 0: équivalent à CrossEntropy
               - γ > 0: réduit la loss pour les exemples bien classés
        alpha: Poids par classe (optionnel)
        label_smoothing: Label smoothing (défaut: 0.0)
    
    Pourquoi c'est utile pour FER2013:
    - La classe "disgust" a très peu d'exemples
    - Les classes "fear" et "sad" sont souvent confondues
    - Focal Loss force le modèle à se concentrer sur ces cas difficiles
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] - Sorties du modèle (avant softmax)
            targets: [B] - Labels (indices de classe)
        
        Returns:
            Loss scalaire
        """
        num_classes = logits.size(1)
        
        # Label smoothing: distribuer un peu de probabilité sur toutes les classes
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # p_t = probabilité de la vraie classe
        pt = (probs * true_dist).sum(dim=1)
        
        # Focal weight: (1 - p_t)^gamma
        # Plus p_t est petit (mal classé), plus le poids est grand
        focal_weight = (1 - pt) ** self.gamma
        
        # Loss avec focal weight
        loss = -(true_dist * log_probs).sum(dim=1) * focal_weight
        
        # Alpha weighting (poids par classe)
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            at = self.alpha[targets]
            loss = loss * at
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    CrossEntropy avec Label Smoothing.
    
    Alternative à nn.CrossEntropyLoss(label_smoothing=...) pour plus de contrôle.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        
        log_probs = F.log_softmax(logits, dim=1)
        
        # Smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -(true_dist * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_class_weights(labels: list, num_classes: int = 7) -> torch.Tensor:
    """
    Calculer les poids de classe pour alpha dans FocalLoss.
    
    Args:
        labels: Liste des labels du dataset
        num_classes: Nombre de classes
    
    Returns:
        Tensor de poids normalisés
    """
    import numpy as np
    from collections import Counter
    
    counter = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)
        # Poids inversement proportionnel à la fréquence
        weight = total / (num_classes * count)
        weights.append(weight)
    
    # Normaliser pour que la moyenne soit 1
    weights = torch.FloatTensor(weights)
    weights = weights / weights.mean()
    
    return weights


# ==================== TEST ====================

if __name__ == "__main__":
    print("Test des fonctions de loss...")
    
    # Données factices
    logits = torch.randn(8, 7)  # Batch de 8, 7 classes
    targets = torch.randint(0, 7, (8,))
    
    # Test FocalLoss
    focal = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss_focal = focal(logits, targets)
    print(f"FocalLoss: {loss_focal.item():.4f}")
    
    # Test avec alpha
    alpha = torch.ones(7)
    alpha[1] = 2.0  # Disgust (classe rare)
    focal_alpha = FocalLoss(gamma=2.0, alpha=alpha, label_smoothing=0.1)
    loss_focal_alpha = focal_alpha(logits, targets)
    print(f"FocalLoss (with alpha): {loss_focal_alpha.item():.4f}")
    
    # Test LabelSmoothingCrossEntropy
    lsce = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_lsce = lsce(logits, targets)
    print(f"LabelSmoothingCE: {loss_lsce.item():.4f}")
    
    # Comparaison avec PyTorch natif
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_ce = ce(logits, targets)
    print(f"PyTorch CE (label_smoothing): {loss_ce.item():.4f}")
    
    # Test get_class_weights
    fake_labels = [0]*100 + [1]*10 + [2]*80 + [3]*200 + [4]*150 + [5]*120 + [6]*90
    weights = get_class_weights(fake_labels, num_classes=7)
    print(f"\nClass weights: {weights}")
    
    print("\n✓ Tous les tests passés!")
