"""
Configuration pour l'entraînement ResNet sur FER2013
=====================================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    """Configuration complète pour l'entraînement."""
    
    # ==================== PATHS ====================
    data_dir: str = "data/fer2013"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    log_dir: str = "logs"
    
    # ==================== DATASET ====================
    num_classes: int = 7
    class_names: List[str] = field(default_factory=lambda: [
        'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
    ])
    img_size: int = 224  # Taille d'entrée pour ResNet (upscale de 48x48)
    original_size: int = 48  # Taille originale FER2013
    
    # ==================== MODEL ====================
    model_name: str = "resnet18"  # resnet18 ou resnet50
    pretrained: bool = True  # Utiliser poids ImageNet
    freeze_backbone: bool = False  # Geler les couches convolutives
    dropout_rate: float = 0.5  # Dropout avant la couche finale
    
    # ==================== TRAINING ====================
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # ==================== AUGMENTATION ====================
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    
    # ==================== HARDWARE ====================
    device: str = "cuda"  # cuda ou cpu
    num_workers: int = 4
    pin_memory: bool = True
    
    # ==================== MISC ====================
    seed: int = 42
    save_every: int = 10  # Sauvegarder tous les N epochs
    log_every: int = 50  # Logger tous les N batches
    
    def __post_init__(self):
        """Créer les dossiers nécessaires."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def get_experiment_name(self) -> str:
        """Générer un nom d'expérience unique."""
        name = f"{self.model_name}"
        name += f"_lr{self.learning_rate}"
        name += f"_bs{self.batch_size}"
        if self.pretrained:
            name += "_pretrained"
        if self.use_augmentation:
            name += "_aug"
        return name


# ==================== CONFIGURATIONS PRÉDÉFINIES ====================

def get_resnet18_config() -> Config:
    """Configuration optimisée pour ResNet-18."""
    return Config(
        model_name="resnet18",
        batch_size=64,
        learning_rate=1e-4,
        num_epochs=100,
        pretrained=True,
        dropout_rate=0.4,
    )


def get_resnet50_config() -> Config:
    """Configuration optimisée pour ResNet-50."""
    return Config(
        model_name="resnet50",
        batch_size=32,  # Plus petit à cause de la mémoire
        learning_rate=5e-5,  # LR plus petit pour modèle plus grand
        num_epochs=100,
        pretrained=True,
        dropout_rate=0.5,
    )


# ==================== NORMALISATION ====================

# Statistiques ImageNet (pour transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Statistiques FER2013 (calculées sur le dataset)
FER2013_MEAN = [0.5071, 0.5071, 0.5071]  # Grayscale répliqué 3 fois
FER2013_STD = [0.2551, 0.2551, 0.2551]


if __name__ == "__main__":
    # Test de la configuration
    config = get_resnet18_config()
    print(f"Experiment: {config.get_experiment_name()}")
    print(f"Model: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Classes: {config.class_names}")
