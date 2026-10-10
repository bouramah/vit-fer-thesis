#!/usr/bin/env python3
"""
📊 Benchmark Comparatif - FER2013 Training
==========================================
Auteur: Ibrahima Douty OULARE

Ce script mesure les performances d'entraînement pour comparer
différents devices (CPU, MPS, CUDA).

Usage:
    python benchmark_training.py --device cuda   # Sur serveur GPU
    python benchmark_training.py --device mps    # Sur Mac
    python benchmark_training.py --device cpu    # Baseline CPU
"""

import argparse
import time
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset


def get_device(device_str):
    """Obtenir le device."""
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple MPS"
    else:
        device = torch.device("cpu")
        name = "CPU"
    
    return device, name


def create_dummy_data(num_samples, batch_size, device):
    """Créer des données factices pour le benchmark."""
    # Simuler des images 224x224 RGB
    X = torch.randn(num_samples, 3, 224, 224)
    y = torch.randint(0, 7, (num_samples,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=0, pin_memory=(device.type == "cuda"))
    
    return loader


def benchmark_resnet(model_name, device, batch_size, num_batches=50):
    """
    Benchmark d'un modèle ResNet.
    
    Returns:
        dict avec les métriques de performance
    """
    # Créer le modèle
    if model_name == "resnet18":
        model = models.resnet18(weights=None, num_classes=7)
    else:
        model = models.resnet50(weights=None, num_classes=7)
    
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Données factices
    X = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 7, (batch_size,), device=device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Synchronisation
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_batches):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    # Synchronisation finale
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    
    total_time = time.time() - start_time
    
    # Calculer les métriques
    time_per_batch = total_time / num_batches * 1000  # ms
    images_per_sec = batch_size * num_batches / total_time
    batches_per_sec = num_batches / total_time
    
    # Estimation pour FER2013 (28709 images train)
    fer_train_size = 28709
    batches_per_epoch = fer_train_size // batch_size
    estimated_epoch_time = batches_per_epoch * (time_per_batch / 1000)  # secondes
    
    return {
        "model": model_name,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "total_time_sec": round(total_time, 3),
        "time_per_batch_ms": round(time_per_batch, 2),
        "images_per_sec": round(images_per_sec, 1),
        "batches_per_sec": round(batches_per_sec, 2),
        "estimated_epoch_time_sec": round(estimated_epoch_time, 1),
        "estimated_50_epochs_min": round(estimated_epoch_time * 50 / 60, 1)
    }


def run_full_benchmark(device, device_name, batch_sizes=[32, 64, 128, 256]):
    """Exécuter le benchmark complet."""
    
    print("\n" + "=" * 70)
    print(f"📊 BENCHMARK DE PERFORMANCE - {device_name}")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "device": str(device),
        "device_name": device_name,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": []
    }
    
    # Ajouter infos GPU si disponible
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        results["gpu_info"] = {
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processors": props.multi_processor_count
        }
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    # Test différents batch sizes
    for model_name in ["resnet18", "resnet50"]:
        print(f"\n{'─' * 50}")
        print(f"🔬 Benchmark {model_name.upper()}")
        print(f"{'─' * 50}")
        
        for bs in batch_sizes:
            # Vérifier si le batch size est réalisable
            try:
                print(f"\n  Batch size {bs}...", end=" ", flush=True)
                
                result = benchmark_resnet(model_name, device, bs)
                results["benchmarks"].append(result)
                
                print(f"✓ {result['images_per_sec']:.0f} img/s, "
                      f"{result['time_per_batch_ms']:.1f}ms/batch, "
                      f"~{result['estimated_50_epochs_min']:.1f}min pour 50 epochs")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"✗ OOM (mémoire insuffisante)")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(f"✗ Erreur: {e}")
                break
    
    return results


def print_comparison_table(results):
    """Afficher un tableau de comparaison."""
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DES PERFORMANCES")
    print("=" * 70)
    
    print(f"\n{'Model':<12} {'Batch':<8} {'Img/s':<10} {'ms/batch':<12} {'~50 epochs':<12}")
    print("-" * 60)
    
    for bench in results["benchmarks"]:
        print(f"{bench['model']:<12} {bench['batch_size']:<8} "
              f"{bench['images_per_sec']:<10.0f} {bench['time_per_batch_ms']:<12.1f} "
              f"{bench['estimated_50_epochs_min']:<12.1f} min")


def main():
    parser = argparse.ArgumentParser(description="Benchmark training performance")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--batch_sizes", type=int, nargs="+", 
                       default=[32, 64, 128, 256])
    parser.add_argument("--output", type=str, default=None,
                       help="Fichier JSON pour sauvegarder les résultats")
    args = parser.parse_args()
    
    # Déterminer le device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    device, device_name = get_device(args.device)
    
    # Adapter les batch sizes selon le device
    if device.type == "cpu":
        batch_sizes = [16, 32]
    elif device.type == "mps":
        batch_sizes = [32, 64, 128]
    else:
        batch_sizes = args.batch_sizes
    
    # Exécuter le benchmark
    results = run_full_benchmark(device, device_name, batch_sizes)
    
    # Afficher le tableau
    print_comparison_table(results)
    
    # Sauvegarder si demandé
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Résultats sauvegardés dans {args.output}")
    
    # Recommandations
    print("\n" + "=" * 70)
    print("🎯 RECOMMANDATION")
    print("=" * 70)
    
    # Trouver le meilleur résultat ResNet-18
    resnet18_results = [b for b in results["benchmarks"] if b["model"] == "resnet18"]
    if resnet18_results:
        best = max(resnet18_results, key=lambda x: x["images_per_sec"])
        print(f"\nMeilleure config ResNet-18: batch_size={best['batch_size']}")
        print(f"  → {best['images_per_sec']:.0f} images/sec")
        print(f"  → ~{best['estimated_50_epochs_min']:.1f} minutes pour 50 epochs")
    
    print("\n")


if __name__ == "__main__":
    main()
