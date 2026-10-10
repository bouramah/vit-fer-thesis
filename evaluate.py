"""
Script d'évaluation ResNet pour FER2013
========================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG

Usage:
    python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_best.pth
    python evaluate.py --model resnet50 --checkpoint checkpoints/resnet50_best.pth
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Imports locaux
from models.resnet import create_resnet18, create_resnet50
from utils.dataset import get_dataloaders, FER2013_CLASSES
from utils.metrics import (
    compute_metrics,
    compute_per_class_metrics,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    print_classification_report
)


def load_model(model_name: str, checkpoint_path: str, device: torch.device):
    """Charger un modèle depuis un checkpoint."""
    
    # Créer le modèle
    if model_name == "resnet18":
        model = create_resnet18(num_classes=7, pretrained=False)
    elif model_name == "resnet50":
        model = create_resnet50(num_classes=7, pretrained=False)
    else:
        raise ValueError(f"Modèle non supporté: {model_name}")
    
    # Charger les poids
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    val_acc = checkpoint.get('val_acc', 'N/A')
    
    print(f"[INFO] Modèle chargé depuis: {checkpoint_path}")
    print(f"[INFO] Epoch: {epoch}, Val Acc: {val_acc:.2f}%")
    
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> tuple:
    """
    Évaluer le modèle sur le test set.
    
    Returns:
        (accuracy, all_preds, all_labels)
    """
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(test_loader, desc="Évaluation")
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'acc': f'{100. * correct / total:.2f}%'})
    
    accuracy = 100. * correct / total
    
    return accuracy, np.array(all_preds), np.array(all_labels)


def main(args):
    """Fonction principale d'évaluation."""
    
    print("\n" + "=" * 70)
    print("📊 ÉVALUATION RESNET SUR FER2013")
    print("=" * 70)
    
    # Device - Support CUDA, MPS (Mac), et CPU
    device_str = args.device
    
    # Auto-détection
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Device: MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("[INFO] Device: CPU")
    
    # Charger le modèle
    print(f"\n[MODEL] Chargement de {args.model}...")
    model = load_model(args.model, args.checkpoint, device)
    model.summary()
    
    # Charger les données
    print(f"\n[DATA] Chargement du test set...")
    _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=224,
        num_workers=args.num_workers,
        use_augmentation=False,  # Pas d'augmentation pour le test
        pin_memory=True
    )
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Évaluation
    print(f"\n[EVAL] Évaluation en cours...")
    accuracy, all_preds, all_labels = evaluate(model, test_loader, device)
    
    # Rapport de classification
    print_classification_report(all_labels, all_preds, args.model)
    
    # Métriques détaillées
    metrics = compute_metrics(all_labels, all_preds)
    per_class = compute_per_class_metrics(all_labels, all_preds)
    
    # Sauvegarder les figures
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Matrice de confusion
    cm_path = os.path.join(args.output_dir, f"{args.model}_confusion_matrix.png")
    plot_confusion_matrix(
        all_labels, all_preds,
        save_path=cm_path,
        title=f"Matrice de Confusion - {args.model.upper()}"
    )
    
    # Accuracy par classe
    acc_path = os.path.join(args.output_dir, f"{args.model}_per_class_accuracy.png")
    per_class_acc = plot_per_class_accuracy(
        all_labels, all_preds,
        save_path=acc_path,
        title=f"Accuracy par Classe - {args.model.upper()}"
    )
    
    # Résumé
    print("\n" + "=" * 70)
    print("📈 RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 70)
    print(f"\n  Modèle: {args.model.upper()}")
    print(f"\n  MÉTRIQUES GLOBALES:")
    print(f"    Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"    Precision: {metrics['precision']:.2f}%")
    print(f"    Recall:    {metrics['recall']:.2f}%")
    print(f"    F1-Score:  {metrics['f1_score']:.2f}%")
    
    print(f"\n  ACCURACY PAR CLASSE:")
    for i, name in enumerate(FER2013_CLASSES):
        print(f"    {name:<10}: {per_class_acc[i]:.1f}%")
    
    print(f"\n  Figures sauvegardées dans: {args.output_dir}")
    print("=" * 70 + "\n")
    
    return metrics


def parse_args():
    """Parser les arguments."""
    parser = argparse.ArgumentParser(
        description="Évaluation ResNet sur FER2013",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet18', 'resnet50'],
                        help='Architecture du modèle')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/fer2013',
                        help='Chemin vers FER2013')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Taille du batch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Nombre de workers')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device (auto, cuda, mps, ou cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Dossier de sortie')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Vérifier les fichiers
    if not os.path.exists(args.data_dir):
        print(f"❌ Dataset non trouvé: {args.data_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint non trouvé: {args.checkpoint}")
        sys.exit(1)
    
    main(args)
