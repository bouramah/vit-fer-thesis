"""
Dataset et DataLoader pour FER2013
===================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


# ==================== CLASSES D'ÉMOTIONS ====================

FER2013_CLASSES = [
    'angry',    # 0 - Colère
    'disgust',  # 1 - Dégoût
    'fear',     # 2 - Peur
    'happy',    # 3 - Joie
    'neutral',  # 4 - Neutre
    'sad',      # 5 - Tristesse
    'surprise'  # 6 - Surprise
]

# Mapping français
FER2013_CLASSES_FR = {
    'angry': 'Colère',
    'disgust': 'Dégoût',
    'fear': 'Peur',
    'happy': 'Joie',
    'neutral': 'Neutre',
    'sad': 'Tristesse',
    'surprise': 'Surprise'
}


# ==================== TRANSFORMATIONS ====================

def get_train_transforms(img_size: int = 224, use_augmentation: bool = True):
    """
    Transformations pour l'entraînement.
    
    Args:
        img_size: Taille de sortie (224 pour ResNet)
        use_augmentation: Activer l'augmentation de données
    
    Returns:
        Composition de transformations
    """
    if use_augmentation:
        return transforms.Compose([
            # Redimensionner avec RandomResizedCrop (meilleur que Resize simple)
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop((img_size, img_size)),
            
            # Convertir en RGB (FER2013 est grayscale)
            transforms.Grayscale(num_output_channels=3),
            
            # Augmentation géométrique
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            
            # Augmentation photométrique (PAS de saturation pour grayscale!)
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0,  # Pas de saturation pour grayscale
            ),
            
            # Random Erasing (simule occlusions) - augmenté
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def get_val_transforms(img_size: int = 224):
    """
    Transformations pour la validation/test (pas d'augmentation).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ==================== SPLIT TRAIN/VAL/TEST ====================

def get_dataloaders_train_val_test(
    data_dir: str,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    use_augmentation: bool = True,
    pin_memory: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Créer les DataLoaders avec split CORRECT: train/val/test.
    
    IMPORTANT: Le test set ne doit être utilisé qu'UNE SEULE FOIS à la fin!
    
    Args:
        data_dir: Chemin vers FER2013
        batch_size: Taille du batch
        img_size: Taille des images
        num_workers: Nombre de workers
        use_augmentation: Activer l'augmentation
        pin_memory: Pin memory pour GPU
        val_ratio: Ratio pour validation (défaut: 10%)
        seed: Random seed pour reproductibilité
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import Subset
    
    # Transformations
    train_transform = get_train_transforms(img_size, use_augmentation)
    val_transform = get_val_transforms(img_size)
    
    # Charger le dataset complet (sans transform pour récupérer les labels)
    full_train = FER2013Dataset(root_dir=data_dir, split="train", transform=None)
    
    # Récupérer les labels pour stratification
    y = np.array([label for _, label in full_train.samples])
    
    # Split stratifié (garde la même proportion de classes dans train et val)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    
    print(f"[DATA] Split stratifié:")
    print(f"  - Train: {len(train_idx)} images ({100*(1-val_ratio):.0f}%)")
    print(f"  - Val:   {len(val_idx)} images ({100*val_ratio:.0f}%)")
    
    # Créer les datasets avec les bonnes transformations
    train_ds = FER2013Dataset(root_dir=data_dir, split="train", transform=train_transform)
    val_ds = FER2013Dataset(root_dir=data_dir, split="train", transform=val_transform)
    test_ds = FER2013Dataset(root_dir=data_dir, split="test", transform=val_transform)
    
    # Appliquer le split
    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)
    
    print(f"  - Test:  {len(test_ds)} images (évaluation finale uniquement)")
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


# ==================== DATASET CLASS ====================

class FER2013Dataset(Dataset):
    """
    Dataset personnalisé pour FER2013.
    
    Supporte:
    - Chargement depuis structure de dossiers (ImageFolder style)
    - Transformations personnalisées
    - Statistiques du dataset
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform = None,
        return_path: bool = False
    ):
        """
        Args:
            root_dir: Chemin vers le dossier FER2013 (contenant train/ et test/)
            split: 'train' ou 'test'
            transform: Transformations à appliquer
            return_path: Retourner aussi le chemin de l'image
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_path = return_path
        
        # Chemin du split
        self.data_dir = self.root_dir / split
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dossier non trouvé: {self.data_dir}\n"
                f"Assurez-vous que FER2013 est téléchargé dans {root_dir}"
            )
        
        # Charger les chemins d'images et labels
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(FER2013_CLASSES)}
        
        for class_name in FER2013_CLASSES:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"[FER2013] Chargé {len(self.samples)} images depuis {split}/")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        
        # Charger l'image
        image = Image.open(img_path)
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Retourner la distribution des classes."""
        labels = [label for _, label in self.samples]
        counter = Counter(labels)
        return {FER2013_CLASSES[k]: v for k, v in sorted(counter.items())}
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculer les poids pour gérer le déséquilibre des classes.
        Poids = total / (num_classes * count_per_class)
        """
        labels = [label for _, label in self.samples]
        counter = Counter(labels)
        total = len(labels)
        num_classes = len(FER2013_CLASSES)
        
        weights = []
        for i in range(num_classes):
            count = counter.get(i, 1)
            weight = total / (num_classes * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def print_statistics(self):
        """Afficher les statistiques du dataset."""
        dist = self.get_class_distribution()
        total = len(self.samples)
        
        print("\n" + "=" * 50)
        print(f"STATISTIQUES FER2013 - {self.split.upper()}")
        print("=" * 50)
        print(f"{'Classe':<12} {'Count':>8} {'Pourcentage':>12}")
        print("-" * 50)
        
        for class_name, count in dist.items():
            pct = (count / total) * 100
            bar = "█" * int(pct / 2)
            print(f"{class_name:<12} {count:>8} {pct:>10.2f}% {bar}")
        
        print("-" * 50)
        print(f"{'TOTAL':<12} {total:>8}")
        print("=" * 50 + "\n")


# ==================== DATA LOADERS ====================

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    use_augmentation: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Créer les DataLoaders pour l'entraînement et le test.
    
    Args:
        data_dir: Chemin vers FER2013
        batch_size: Taille du batch
        img_size: Taille des images
        num_workers: Nombre de workers pour le chargement
        use_augmentation: Activer l'augmentation
        pin_memory: Activer pin_memory pour GPU
    
    Returns:
        (train_loader, test_loader)
    """
    # Transformations
    train_transform = get_train_transforms(img_size, use_augmentation)
    test_transform = get_val_transforms(img_size)
    
    # Datasets
    train_dataset = FER2013Dataset(
        root_dir=data_dir,
        split="train",
        transform=train_transform
    )
    
    test_dataset = FER2013Dataset(
        root_dir=data_dir,
        split="test",
        transform=test_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Éviter les batches de taille variable
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_sample_batch(data_dir: str, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Obtenir un batch d'exemple pour les tests et visualisations.
    """
    transform = get_val_transforms(224)
    dataset = FER2013Dataset(data_dir, split="train", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))
    return images, labels


# ==================== TEST ====================

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("TEST DU MODULE DATASET FER2013")
    print("=" * 60)
    
    # Chemin du dataset (à modifier selon votre configuration)
    DATA_DIR = "data/fer2013"
    
    if not os.path.exists(DATA_DIR):
        print(f"\n⚠️  Dataset non trouvé dans: {DATA_DIR}")
        print("Téléchargez FER2013 depuis Kaggle:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        print("\nStructure attendue:")
        print(f"{DATA_DIR}/")
        print("├── train/")
        print("│   ├── angry/")
        print("│   ├── disgust/")
        print("│   └── ...")
        print("└── test/")
        print("    ├── angry/")
        print("    └── ...")
        sys.exit(1)
    
    # Test du dataset
    print("\n[TEST] Chargement du dataset...")
    train_dataset = FER2013Dataset(DATA_DIR, split="train", transform=get_train_transforms())
    test_dataset = FER2013Dataset(DATA_DIR, split="test", transform=get_val_transforms())
    
    # Statistiques
    train_dataset.print_statistics()
    test_dataset.print_statistics()
    
    # Test d'un sample
    print("[TEST] Chargement d'un sample...")
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label} ({FER2013_CLASSES[label]})")
    
    # Test des DataLoaders
    print("\n[TEST] Création des DataLoaders...")
    train_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=32)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test d'un batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Poids de classe
    print("\n[TEST] Calcul des poids de classe...")
    weights = train_dataset.get_class_weights()
    print("Class weights:")
    for i, (name, w) in enumerate(zip(FER2013_CLASSES, weights)):
        print(f"  {name}: {w:.4f}")
    
    print("\n" + "=" * 60)
    print("TOUS LES TESTS PASSÉS ✓")
    print("=" * 60)
