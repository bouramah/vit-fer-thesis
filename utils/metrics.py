"""
Métriques d'évaluation pour FER2013
====================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# Classes FER2013
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASS_NAMES_FR = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Neutre', 'Tristesse', 'Surprise']


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculer toutes les métriques de classification.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        average: Type de moyenne ('weighted', 'macro', 'micro')
    
    Returns:
        Dictionnaire avec accuracy, precision, recall, f1
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0) * 100,
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0) * 100,
    }
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Calculer les métriques par classe.
    
    Returns:
        Dictionnaire {classe: {precision, recall, f1}}
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1_score': f1[i] * 100
        }
    
    return per_class


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculer la matrice de confusion.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        normalize: Normaliser les valeurs (pourcentages)
    
    Returns:
        Matrice de confusion numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Remplacer NaN par 0
    
    return cm


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Matrice de Confusion",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True
):
    """
    Tracer et sauvegarder la matrice de confusion.
    """
    cm = get_confusion_matrix(y_true, y_pred, normalize=normalize)
    
    plt.figure(figsize=figsize)
    
    # Formatage selon normalisation
    fmt = '.2f' if normalize else 'd'
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=CLASS_NAMES_FR,
        yticklabels=CLASS_NAMES_FR,
        vmin=0,
        vmax=1 if normalize else None
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matrice de confusion sauvegardée: {save_path}")
    
    plt.close()
    return cm


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None
):
    """
    Tracer les courbes d'entraînement (loss et accuracy).
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Courbe de Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Courbe d'Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Courbes d'entraînement sauvegardées: {save_path}")
    
    plt.close()


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Accuracy par Classe"
):
    """
    Tracer un bar chart de l'accuracy par classe.
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(CLASS_NAMES)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(CLASS_NAMES_FR, per_class_acc, color=colors, edgecolor='black', linewidth=1.2)
    
    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars, per_class_acc):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{acc:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.xlabel('Émotion', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy par classe sauvegardée: {save_path}")
    
    plt.close()
    return per_class_acc


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
):
    """
    Afficher un rapport de classification détaillé.
    """
    print("\n" + "=" * 70)
    print(f"RAPPORT DE CLASSIFICATION - {model_name.upper()}")
    print("=" * 70)
    
    # Métriques globales
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n📊 MÉTRIQUES GLOBALES (weighted)")
    print("-" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']:.2f}%")
    
    # Métriques par classe
    print(f"\n📈 MÉTRIQUES PAR CLASSE")
    print("-" * 70)
    print(f"{'Classe':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 70)
    
    per_class = compute_per_class_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    supports = cm.sum(axis=1)
    
    for i, name in enumerate(CLASS_NAMES):
        metrics_class = per_class[name]
        print(f"{name:<12} {metrics_class['precision']:>11.2f}% {metrics_class['recall']:>11.2f}% "
              f"{metrics_class['f1_score']:>11.2f}% {supports[i]:>10}")
    
    print("-" * 70)
    print(f"{'Total':<12} {'':<12} {'':<12} {'':<12} {sum(supports):>10}")
    print("=" * 70 + "\n")
    
    return metrics


class MetricTracker:
    """
    Classe pour suivre les métriques pendant l'entraînement.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Réinitialiser toutes les métriques."""
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float
    ):
        """Mettre à jour les métriques pour une epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            return True  # Nouveau meilleur modèle
        
        return False
    
    def get_summary(self) -> Dict:
        """Obtenir un résumé des métriques."""
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_train_acc': self.train_accs[-1] if self.train_accs else 0,
            'final_val_acc': self.val_accs[-1] if self.val_accs else 0,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
        }
    
    def save_plots(self, save_dir: str, model_name: str):
        """Sauvegarder les courbes d'entraînement."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_accs,
            self.val_accs,
            save_path
        )


if __name__ == "__main__":
    # Test des métriques
    print("Test des métriques...")
    
    # Données simulées
    np.random.seed(42)
    y_true = np.random.randint(0, 7, 1000)
    y_pred = np.random.randint(0, 7, 1000)
    
    # Test des métriques
    metrics = compute_metrics(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    
    # Test du rapport
    print_classification_report(y_true, y_pred, "Test Model")
    
    # Test du tracker
    tracker = MetricTracker()
    for epoch in range(10):
        tracker.update(epoch, 1.0 - epoch*0.05, 1.2 - epoch*0.04, 
                      50 + epoch*3, 45 + epoch*2.5)
    
    print(f"Best accuracy: {tracker.best_val_acc:.2f}% at epoch {tracker.best_epoch}")
    print("\n✓ Tous les tests passés!")
