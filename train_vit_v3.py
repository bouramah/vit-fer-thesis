"""
Script d'entraînement Vision Transformer (ViT) pour FER2013 - VERSION 3
========================================================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG

CORRECTIONS v3 (basées sur analyse ChatGPT):
✅ Fix: Métriques train désactivées quand mixup ON (évite bruit)
✅ Fix: Pas de combo sampler + focal + mixup (sur-pondération)
✅ Fix: Régularisation équilibrée (dropout 0.2, drop_path 0.1-0.15)
✅ Fix: Mixup réduit à 0.1, label_smoothing à 0.05
✅ Ajout: Monitoring gap train/val amélioré

CONFIGS RECOMMANDÉES:
    # ViT-Small (meilleur)
    python train_vit_v3.py --model vit_small --epochs 70 --batch_size 64 \\
        --lr 1e-4 --weight_decay 0.05 --drop_path 0.1 --dropout 0.2 \\
        --mixup 0.1 --label_smoothing 0.05 --weighted_sampler

    # DeiT-Small
    python train_vit_v3.py --model deit_small --epochs 70 --batch_size 64 \\
        --lr 1e-4 --weight_decay 0.05 --drop_path 0.15 --dropout 0.2 \\
        --mixup 0.1 --label_smoothing 0.05 --weighted_sampler
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

# Imports locaux
from utils.dataset import (
    FER2013Dataset, 
    get_train_transforms, 
    get_val_transforms,
    FER2013_CLASSES
)
from utils.losses import FocalLoss, get_class_weights
from utils.metrics import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    print_classification_report,
    plot_training_curves
)


def set_seed(seed: int = 42):
    """Fixer les seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_str: str = "auto") -> torch.device:
    """Obtenir le device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] GPU NVIDIA: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] GPU Apple (MPS)")
    else:
        device = torch.device("cpu")
        print("[INFO] CPU")
    
    return device


# ==================== WEIGHTED SAMPLER ====================

def make_weighted_sampler(dataset, indices=None):
    """
    Créer un WeightedRandomSampler pour équilibrer les classes.
    """
    if indices is not None:
        labels = np.array([dataset.samples[i][1] for i in indices])
    else:
        labels = np.array([label for _, label in dataset.samples])
    
    class_counts = np.bincount(labels, minlength=7)
    
    # Poids inversement proportionnel au nombre d'échantillons
    class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-6)
    
    # Normaliser
    class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-6)

    
    # Assigner un poids à chaque échantillon
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True
    )

    
    print(f"[INFO] WeightedRandomSampler créé")
    print(f"  Class counts: {class_counts}")
    print(f"  Class weights: {np.round(class_weights, 4)}")
    
    return sampler


# ==================== DATA LOADING ====================

def get_dataloaders_v3(
    data_dir: str,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    use_augmentation: bool = True,
    pin_memory: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_weighted_sampler: bool = False
):
    """
    Créer les DataLoaders avec options améliorées.
    """
    train_transform = get_train_transforms(img_size, use_augmentation)
    val_transform = get_val_transforms(img_size)
    
    # Charger le dataset complet
    full_train = FER2013Dataset(root_dir=data_dir, split="train", transform=None)
    
    # Labels pour stratification
    y = np.array([label for _, label in full_train.samples])
    
    # Split stratifié
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    
    print(f"[DATA] Split stratifié:")
    print(f"  - Train: {len(train_idx)} images ({100*(1-val_ratio):.0f}%)")
    print(f"  - Val:   {len(val_idx)} images ({100*val_ratio:.0f}%)")
    
    # Datasets avec transformations
    train_ds_full = FER2013Dataset(root_dir=data_dir, split="train", transform=train_transform)
    val_ds_full = FER2013Dataset(root_dir=data_dir, split="train", transform=val_transform)
    test_ds = FER2013Dataset(root_dir=data_dir, split="test", transform=val_transform)
    
    # Subsets
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)
    
    print(f"  - Test:  {len(test_ds)} images")
    
    # Train DataLoader
    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds_full, train_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    else:
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
    
    # Poids de classe pour Focal Loss (si utilisé seul)
    train_labels = [full_train.samples[i][1] for i in train_idx]
    class_weights = get_class_weights(train_labels, num_classes=7)
    
    return train_loader, val_loader, test_loader, class_weights


# ==================== MODEL CREATION ====================

def create_model_v3(
    model_name: str,
    pretrained: bool = True,
    dropout: float = 0.2,
    drop_path_rate: float = 0.1
):
    """
    Créer le modèle ViT avec drop_path.
    """
    from models.vit import ViTForFER
    
    timm_models = {
        'vit_tiny': 'vit_tiny_patch16_224',
        'vit_small': 'vit_small_patch16_224',
        'vit_base': 'vit_base_patch16_224',
        'deit_tiny': 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base': 'deit_base_patch16_224',
    }
    
    if model_name not in timm_models:
        raise ValueError(f"Modèle non supporté: {model_name}")
    
    print(f"[MODEL] Chargement de {model_name}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Dropout: {dropout}")
    print(f"  Drop path rate: {drop_path_rate}")
    
    model = ViTForFER(
        model_name=timm_models[model_name],
        num_classes=7,
        pretrained=pretrained,
        dropout=dropout,
        drop_path_rate=drop_path_rate
    )
    
    return model


# ==================== MIXUP ====================

def mixup_data(x, y, alpha=0.1):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss pour mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== TRAINING (CORRIGÉ v3) ====================

def train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch,
    use_mixup=False, mixup_alpha=0.1
):
    """
    Entraîner pour une epoch.
    
    CORRECTION v3: Quand mixup est activé, on ne calcule PAS les métriques
    train (F1, accuracy) car elles seraient bruitées et trompeuses.
    """
    model.train()
    running_loss = 0.0
    
    # Métriques train (fiables seulement sans mixup)
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_mixup:
            # Avec mixup: mélange des images et labels
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # ⚠️ NE PAS calculer F1 train avec mixup (trop bruité!)
            # Les métriques seraient fausses car les labels sont mélangés
            
        else:
            # Sans mixup: entraînement normal
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Collecter les prédictions pour les métriques
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{running_loss / (pbar.n + 1):.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    
    # Si mixup activé, retourner None pour les métriques
    if use_mixup:
        return epoch_loss, None, None
    
    # Sinon, calculer les vraies métriques
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    epoch_acc = 100.0 * np.mean(all_preds == all_labels)
    epoch_f1 = 100.0 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch):
    """Valider le modèle."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_acc = 100. * np.mean(all_preds == all_labels)
    val_f1 = 100. * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return val_loss, val_acc, val_f1, all_preds, all_labels


def save_checkpoint(model, optimizer, scheduler, epoch, val_f1, save_path, is_best=False):
    """Sauvegarder un checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_f1': val_f1,
    }
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


# ==================== MAIN ====================

def main(args):
    """Fonction principale."""
    
    print("\n" + "=" * 70)
    print("🎭 ENTRAÎNEMENT VISION TRANSFORMER v3 - FER2013")
    print("=" * 70)
    
    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    set_seed(args.seed)
    device = get_device(args.device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model}_v3_{timestamp}"
    
    print(f"\n[CONFIG]")
    print(f"  Modèle: {args.model}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Drop path rate: {args.drop_path}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Mixup: {args.mixup}")
    print(f"  Focal Loss: {args.focal_loss}")
    print(f"  Weighted Sampler: {args.weighted_sampler}")
    
    # ==================== DATA ====================
    print("\n[DATA] Chargement...")
    
    train_loader, val_loader, test_loader, class_weights = get_dataloaders_v3(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=224,
        num_workers=args.num_workers,
        use_augmentation=True,
        pin_memory=True,
        val_ratio=0.1,
        seed=args.seed,
        use_weighted_sampler=args.weighted_sampler
    )
    
    # ==================== MODEL ====================
    model = create_model_v3(
        args.model,
        pretrained=args.pretrained,
        dropout=args.dropout,
        drop_path_rate=args.drop_path
    )
    model = model.to(device)
    model.summary()
    
    # ==================== LOSS (CORRIGÉ v3) ====================
    # RÈGLE: Ne pas combiner weighted_sampler + focal_loss + mixup
    # Ça sur-pondère les classes rares 2 fois!
    
    if args.focal_loss and (args.weighted_sampler or args.mixup > 0):
        print("\n[LOSS] ⚠️ FocalLoss désactivée (sampler/mixup actif)")
        print("       → Utilisation de CrossEntropyLoss + label_smoothing")
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.focal_loss:
        print("\n[LOSS] FocalLoss (gamma=2.0)")
        alpha = class_weights.to(device) if args.use_class_weights else None
        criterion = FocalLoss(
            gamma=2.0,
            alpha=alpha,
            label_smoothing=args.label_smoothing
        )
    else:
        print("\n[LOSS] CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # ==================== OPTIMIZER ====================
    if hasattr(model, 'backbone') and hasattr(model, 'classifier'):
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': args.lr * 0.1, 'name': 'backbone'},
            {'params': model.classifier.parameters(), 'lr': args.lr, 'name': 'classifier'}
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"[OPTIMIZER] AdamW avec LR différentiel")
        print(f"  Backbone LR: {args.lr * 0.1:.2e}")
        print(f"  Classifier LR: {args.lr:.2e}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"[OPTIMIZER] AdamW, LR: {args.lr:.2e}")
    
    # ==================== SCHEDULER ====================
    warmup_epochs = min(5, args.epochs // 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - warmup_epochs, 
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    
    print(f"[SCHEDULER] Warmup {warmup_epochs} epochs + Cosine Annealing (min_lr=1e-6)")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, exp_name))
    
    # Tracking
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    use_mixup = args.mixup > 0
    
    # ==================== TRAINING LOOP ====================
    print("\n" + "=" * 70)
    print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
    if use_mixup:
        print("   ⚠️ Mixup activé: métriques train indisponibles")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mixup=use_mixup, mixup_alpha=args.mixup
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        scheduler.step()
        
        # Learning rates
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        
        # Tracking
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print (adapté pour mixup)
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        
        if train_acc is None:
            # Mixup activé: pas de métriques train fiables
            print(f"  Train - Loss: {train_loss:.4f} (mixup ON, métriques N/A)")
            train_accs.append(np.nan)  # Placeholder
        else:
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%")
            train_accs.append(train_acc)
            
            # Overfit gap (seulement si on a les métriques train)
            gap_acc = train_acc - val_acc
            gap_f1 = train_f1 - val_f1
            print(f"  Overfit gap - Acc: {gap_acc:.2f}%, F1: {gap_f1:.2f}%")
            writer.add_scalars('Overfit_Gap', {'acc': gap_acc, 'f1': gap_f1}, epoch)
        
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%")
        val_accs.append(val_acc)
        
        # Afficher LRs
        if len(lrs) > 1:
            print(f"  LR backbone: {lrs[0]:.2e} | LR head: {lrs[1]:.2e}")
        else:
            print(f"  LR: {lrs[0]:.2e}")
        
        # Best model (basé sur Val F1 uniquement!)
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            print(f"  ✓ Nouveau meilleur modèle! (F1: {val_f1:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Macro-F1/val', val_f1, epoch)
        
        if train_acc is not None:
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Macro-F1/train', train_f1, epoch)
        
        for i, lr in enumerate(lrs):
            writer.add_scalar(f'LR/param_group_{i}', lr, epoch)
        
        # Checkpoint
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_path = os.path.join(args.checkpoint_dir, f"{args.model}_v3_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_f1, save_path, is_best)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[EARLY STOPPING] Pas d'amélioration depuis {args.patience} epochs")
            break
    
    total_time = time.time() - start_time
    
    # ==================== FINAL EVALUATION ====================
    print("\n" + "=" * 70)
    print("📊 ÉVALUATION FINALE SUR LE TEST SET")
    print("=" * 70)
    
    # Charger le meilleur modèle
    best_path = os.path.join(args.checkpoint_dir, f"{args.model}_v3_epoch{best_epoch+1}_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Meilleur modèle chargé (epoch {best_epoch+1}, Val F1: {best_val_f1:.2f}%)")
    
    # Test
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device, 0
    )
    
    print(f"\n🎯 RÉSULTATS SUR TEST SET:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Macro-F1: {test_f1:.2f}%")
    
    # Rapport
    print_classification_report(test_labels, test_preds, args.model)
    
    # Figures
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=os.path.join(args.results_dir, f"{exp_name}_confusion_matrix.png"),
        title=f"Matrice de Confusion - {args.model.upper()} v3 (Test)"
    )
    
    plot_per_class_accuracy(
        test_labels, test_preds,
        save_path=os.path.join(args.results_dir, f"{exp_name}_per_class_accuracy.png"),
        title=f"Accuracy par Classe - {args.model.upper()} v3 (Test)"
    )
    
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(args.results_dir, f"{exp_name}_training_curves.png")
    )
    
    # Résumé
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"  Modèle: {args.model} v3")
    print(f"  Meilleur Val F1: {best_val_f1:.2f}% (epoch {best_epoch+1})")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Macro-F1: {test_f1:.2f}%")
    print(f"  Temps total: {total_time/60:.1f} minutes")
    print("=" * 70 + "\n")
    
    writer.close()
    
    return test_f1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraînement ViT v3 pour FER2013 (optimisé)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    parser.add_argument('--model', type=str, default='vit_small',
                       choices=['vit_tiny', 'vit_small', 'vit_base', 
                               'deit_tiny', 'deit_small', 'deit_base'])
    
    # Pretrained
    parser.add_argument('--pretrained', action='store_true',
                       help='Utiliser les poids ImageNet')
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Ne pas utiliser de poids pré-entraînés')
    
    # Regularization (valeurs optimisées v3)
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout (0.2 recommandé)')
    parser.add_argument('--drop_path', type=float, default=0.1,
                       help='Drop path rate (0.1 vit, 0.15 deit)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/fer2013')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=70,
                       help='Nombre d\'epochs (70+ recommandé)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay (0.05 recommandé pour ViT)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Loss & Sampling
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                       help='Label smoothing (0.05 recommandé)')
    parser.add_argument('--focal_loss', action='store_true',
                       help='Utiliser Focal Loss (désactivé si sampler/mixup)')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Utiliser les poids de classe dans Focal Loss')
    parser.add_argument('--weighted_sampler', action='store_true',
                       help='Utiliser WeightedRandomSampler')
    
    # Augmentation
    parser.add_argument('--mixup', type=float, default=0.1,
                       help='Mixup alpha (0.1 recommandé, 0 = désactivé)')
    
    # Misc
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=10)
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_vit_v3')
    parser.add_argument('--results_dir', type=str, default='results_vit_v3')
    parser.add_argument('--log_dir', type=str, default='logs_vit_v3')
    
    args = parser.parse_args()
    
    # Gérer pretrained
    if args.no_pretrained:
        args.pretrained = False
    else:
        args.pretrained = True
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"\n❌ Dataset non trouvé: {args.data_dir}")
        sys.exit(1)
    
    # Vérifier timm
    try:
        import timm
        print(f"[INFO] timm version: {timm.__version__}")
    except ImportError:
        print("⚠️ timm non installé. Installation...")
        os.system("pip install timm")
        import timm
    
    main(args)