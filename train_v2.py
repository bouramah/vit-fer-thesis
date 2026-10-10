"""
Script d'entraînement ResNet pour FER2013 - VERSION 2.0 AMÉLIORÉE
==================================================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG

AMÉLIORATIONS v2:
✅ Split correct train/val/test (pas de data leakage)
✅ Macro-F1 comme métrique principale
✅ Label Smoothing
✅ Warmup scheduler
✅ Early stopping sur Macro-F1
✅ Mixup data augmentation (optionnel)
✅ Two-stage fine-tuning (optionnel)
✅ Support MPS (Mac) + CUDA + CPU

Usage:
    python train_v2.py --model resnet18 --epochs 50
    python train_v2.py --model resnet50 --epochs 50 --mixup 0.2
    python train_v2.py --model resnet18 --two_stage --epochs 50
"""

import os
import sys
import argparse
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

# Imports locaux
from models.resnet import create_resnet18, create_resnet50
from utils.dataset import get_dataloaders_train_val_test, FER2013_CLASSES, FER2013Dataset
from utils.metrics import (
    compute_metrics,
    MetricTracker,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    print_classification_report,
    plot_training_curves
)


def set_seed(seed: int = 42):
    """Fixer les seeds pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: deterministic=True ralentit l'entraînement
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Plus rapide sur GPU
    print(f"[INFO] Seed fixé à {seed}")


def get_device(device_str: str = "auto") -> torch.device:
    """Obtenir le device (GPU ou CPU)."""
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


def create_model(model_name: str, pretrained: bool = True, dropout: float = 0.5):
    """Créer le modèle."""
    if model_name == "resnet18":
        return create_resnet18(num_classes=7, pretrained=pretrained, dropout_rate=dropout)
    elif model_name == "resnet50":
        return create_resnet50(num_classes=7, pretrained=pretrained, dropout_rate=dropout)
    else:
        raise ValueError(f"Modèle non supporté: {model_name}")


# ==================== MIXUP AUGMENTATION ====================

def mixup_data(x, y, alpha=0.2):
    """
    Mixup: mélange deux images et leurs labels.
    
    Mixup crée des exemples virtuels en combinant linéairement
    deux images et leurs labels one-hot.
    
    Avantages:
    - Régularisation forte
    - Réduit l'overfitting
    - Améliore la généralisation
    """
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


# ==================== TRAINING FUNCTIONS ====================

def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2
) -> tuple:
    """Entraîner pour une epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_mixup:
            # Mixup augmentation
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # Pour les métriques, on utilise le label dominant
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            # Utiliser le label avec le plus grand poids
            effective_labels = labels_a if lam > 0.5 else labels_b
            all_labels.extend(effective_labels.cpu().numpy())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{running_loss / (pbar.n + 1):.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    
    # Calculer accuracy et macro-f1
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    epoch_acc = 100. * np.mean(all_preds == all_labels)
    epoch_f1 = 100. * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
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


# ==================== TWO-STAGE FINE-TUNING ====================

def freeze_backbone(model):
    """Geler le backbone (seulement classifier entraînable)."""
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'fc' not in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Backbone gelé: {trainable:,} / {total:,} paramètres entraînables")


def unfreeze_backbone(model):
    """Dégeler tout le modèle."""
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Backbone dégelé: {trainable:,} paramètres entraînables")


# ==================== MAIN ====================

def main(args):
    """Fonction principale."""
    
    print("\n" + "=" * 70)
    print("🎭 ENTRAÎNEMENT RESNET v2.0 - FER2013")
    print("=" * 70)
    
    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    set_seed(args.seed)
    device = get_device(args.device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model}_v2_{timestamp}"
    
    print(f"\n[CONFIG]")
    print(f"  Modèle: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Mixup: {args.mixup}")
    print(f"  Two-stage: {args.two_stage}")
    print(f"  Dropout: {args.dropout}")
    
    # ==================== DATA ====================
    print("\n[DATA] Chargement avec split train/val/test...")
    
    train_loader, val_loader, test_loader = get_dataloaders_train_val_test(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=224,
        num_workers=args.num_workers,
        use_augmentation=True,
        pin_memory=True,
        val_ratio=0.1,
        seed=args.seed
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)} (utilisé à la fin uniquement)")
    
    # ==================== MODEL ====================
    print(f"\n[MODEL] Création de {args.model}...")
    
    model = create_model(args.model, pretrained=True, dropout=args.dropout)
    model = model.to(device)
    model.summary()
    
    # ==================== LOSS ====================
    # Label smoothing aide à réduire l'overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # ==================== TWO-STAGE TRAINING ====================
    if args.two_stage:
        print("\n[TWO-STAGE] Phase 1: Entraînement du classifier uniquement")
        freeze_backbone(model)
        
        # Phase 1: Juste le classifier avec LR plus élevé
        optimizer_stage1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 10,  # LR plus élevé pour le classifier
            weight_decay=args.weight_decay
        )
        
        # Entraîner le classifier pendant quelques epochs
        stage1_epochs = min(5, args.epochs // 5)
        for epoch in range(stage1_epochs):
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer_stage1, device, epoch,
                use_mixup=args.mixup > 0, mixup_alpha=args.mixup
            )
            val_loss, val_acc, val_f1, _, _ = validate(
                model, val_loader, criterion, device, epoch
            )
            print(f"Stage1 Epoch {epoch+1}/{stage1_epochs} - "
                  f"Train F1: {train_f1:.2f}%, Val F1: {val_f1:.2f}%")
        
        print("\n[TWO-STAGE] Phase 2: Fine-tuning complet")
        unfreeze_backbone(model)
    
    # ==================== OPTIMIZER & SCHEDULER ====================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Warmup + Cosine Annealing
    warmup_epochs = min(5, args.epochs // 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    print(f"\n[SCHEDULER] Warmup {warmup_epochs} epochs + Cosine Annealing")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, exp_name))
    
    # Tracking
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Pour les courbes
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    
    # ==================== TRAINING LOOP ====================
    print("\n" + "=" * 70)
    print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mixup=args.mixup > 0, mixup_alpha=args.mixup
        )
        
        # Validate (sur val_loader, PAS test_loader!)
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Tracking
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%")
        print(f"  LR: {current_lr:.2e}")
        
        # Check if best (basé sur F1, pas accuracy!)
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
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalars('Macro-F1', {'train': train_f1, 'val': val_f1}, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_path = os.path.join(args.checkpoint_dir, f"{args.model}_v2_epoch{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_f1, save_path, is_best)
            if is_best:
                print(f"  ✓ Checkpoint sauvegardé: {save_path}")
        
        # Early stopping (basé sur F1!)
        if patience_counter >= args.patience:
            print(f"\n[EARLY STOPPING] Pas d'amélioration du F1 depuis {args.patience} epochs")
            break
    
    total_time = time.time() - start_time
    
    # ==================== FINAL EVALUATION ON TEST SET ====================
    print("\n" + "=" * 70)
    print("📊 ÉVALUATION FINALE SUR LE TEST SET")
    print("=" * 70)
    
    # Charger le meilleur modèle
    best_path = os.path.join(args.checkpoint_dir, f"{args.model}_v2_epoch{best_epoch+1}_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Meilleur modèle chargé (epoch {best_epoch+1}, Val F1: {best_val_f1:.2f}%)")
    
    # Évaluation sur TEST set (première et unique fois!)
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device, 0
    )
    
    print(f"\n🎯 RÉSULTATS SUR TEST SET:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Macro-F1: {test_f1:.2f}%")
    
    # Rapport détaillé
    print_classification_report(test_labels, test_preds, args.model)
    
    # Sauvegarder les figures
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=os.path.join(args.results_dir, f"{exp_name}_confusion_matrix.png"),
        title=f"Matrice de Confusion - {args.model.upper()} (Test)"
    )
    
    plot_per_class_accuracy(
        test_labels, test_preds,
        save_path=os.path.join(args.results_dir, f"{exp_name}_per_class_accuracy.png"),
        title=f"Accuracy par Classe - {args.model.upper()} (Test)"
    )
    
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(args.results_dir, f"{exp_name}_training_curves.png")
    )
    
    # Résumé final
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"  Modèle: {args.model}")
    print(f"  Meilleur Val F1: {best_val_f1:.2f}% (epoch {best_epoch+1})")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test Macro-F1: {test_f1:.2f}%")
    print(f"  Temps total: {total_time/60:.1f} minutes")
    print(f"  Résultats: {args.results_dir}")
    print("=" * 70 + "\n")
    
    writer.close()
    
    return test_f1


def parse_args():
    """Parser les arguments."""
    parser = argparse.ArgumentParser(
        description="Entraînement ResNet v2.0 pour FER2013",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'])
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/fer2013')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    
    # Améliorations v2
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing (0 = désactivé)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='Mixup alpha (0 = désactivé, recommandé: 0.2)')
    parser.add_argument('--two_stage', action='store_true',
                        help='Two-stage fine-tuning (geler backbone puis dégeler)')
    
    # Misc
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=10)
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--log_dir', type=str, default='logs')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"\n❌ Dataset non trouvé: {args.data_dir}")
        print("Téléchargez FER2013 depuis Kaggle")
        sys.exit(1)
    
    main(args)
