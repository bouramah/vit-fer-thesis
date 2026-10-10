#!/usr/bin/env python3
"""
🔥 Vérification GPU NVIDIA A100 pour FER2013
=============================================
Auteur: Ibrahima Douty OULARE

Ce script vérifie que PyTorch peut utiliser les GPUs A100
et fait un benchmark de performance.
"""

import sys
import time

def check_pytorch():
    """Vérifier PyTorch et CUDA."""
    print("\n" + "=" * 60)
    print("🔥 VÉRIFICATION PYTORCH + CUDA")
    print("=" * 60)
    
    try:
        import torch
        print(f"\n✅ PyTorch installé: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch non installé!")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return None
    
    # CUDA
    print(f"\n📌 CUDA:")
    print(f"   Disponible: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible!")
        return None
    
    print(f"   Version CUDA: {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    return torch


def check_gpus(torch):
    """Lister les GPUs disponibles."""
    print("\n" + "=" * 60)
    print("🎮 GPUs DISPONIBLES")
    print("=" * 60)
    
    num_gpus = torch.cuda.device_count()
    print(f"\n📌 Nombre de GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / 1e9
        mem_used = torch.cuda.memory_allocated(i) / 1e9
        mem_free = mem_total - mem_used
        
        print(f"\n   GPU {i}: {props.name}")
        print(f"      Compute Capability: {props.major}.{props.minor}")
        print(f"      Mémoire totale: {mem_total:.1f} GB")
        print(f"      Mémoire libre: {mem_free:.1f} GB")
        print(f"      Multi-processeurs: {props.multi_processor_count}")
    
    return num_gpus


def benchmark_gpu(torch, gpu_id=0):
    """Benchmark de performance GPU."""
    print("\n" + "=" * 60)
    print(f"⚡ BENCHMARK GPU {gpu_id}")
    print("=" * 60)
    
    device = torch.device(f"cuda:{gpu_id}")
    print(f"\n📌 Device: {torch.cuda.get_device_name(gpu_id)}")
    
    # Test 1: Multiplication matricielle
    print("\n   Test 1: Multiplication matricielle 10000×10000")
    size = 10000
    iterations = 10
    
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    _ = torch.mm(a, b)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()
    matmul_time = time.time() - start
    
    tflops = (2 * size**3 * iterations) / matmul_time / 1e12
    print(f"      Temps: {matmul_time:.3f}s ({iterations} itérations)")
    print(f"      Performance: {tflops:.2f} TFLOPS")
    
    # Test 2: Convolution (comme dans ResNet)
    print("\n   Test 2: Convolution 3D (batch=64, 224×224)")
    import torch.nn as nn
    
    conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
    x = torch.randn(64, 64, 224, 224, device=device)
    
    # Warmup
    _ = conv(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = conv(x)
    torch.cuda.synchronize()
    conv_time = time.time() - start
    
    print(f"      Temps: {conv_time:.3f}s ({iterations} itérations)")
    print(f"      Temps/batch: {conv_time/iterations*1000:.1f}ms")
    
    # Test 3: ResNet-18 forward pass
    print("\n   Test 3: ResNet-18 inference (batch=64)")
    from torchvision import models
    
    model = models.resnet18(weights=None).to(device)
    model.eval()
    x = torch.randn(64, 3, 224, 224, device=device)
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
    torch.cuda.synchronize()
    resnet_time = time.time() - start
    
    images_per_sec = 64 * iterations / resnet_time
    print(f"      Temps: {resnet_time:.3f}s ({iterations} itérations)")
    print(f"      Throughput: {images_per_sec:.0f} images/sec")
    
    return images_per_sec


def recommend_settings(torch, num_gpus):
    """Recommandations pour l'entraînement."""
    print("\n" + "=" * 60)
    print("📋 RECOMMANDATIONS POUR FER2013")
    print("=" * 60)
    
    # Mémoire GPU
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if mem >= 40:  # A100 40GB
        batch_size = 256
        num_workers = 8
    elif mem >= 16:  # V100, RTX 3090
        batch_size = 128
        num_workers = 4
    else:
        batch_size = 64
        num_workers = 4
    
    print(f"""
📌 Configuration recommandée:

   # Utiliser un seul GPU (suffisant pour FER2013)
   python train_v2.py \\
       --model resnet18 \\
       --device cuda \\
       --batch_size {batch_size} \\
       --num_workers {num_workers} \\
       --epochs 50 \\
       --mixup 0.2

   # Pour ResNet-50 (plus de mémoire)
   python train_v2.py \\
       --model resnet50 \\
       --device cuda \\
       --batch_size {batch_size // 2} \\
       --num_workers {num_workers} \\
       --epochs 50 \\
       --mixup 0.2 \\
       --two_stage

📌 Pour utiliser un GPU spécifique:
   CUDA_VISIBLE_DEVICES=1 python train_v2.py --model resnet18

📌 Temps estimé sur A100:
   - ResNet-18: ~5-10 minutes (50 epochs)
   - ResNet-50: ~15-20 minutes (50 epochs)
""")


def main():
    """Fonction principale."""
    print("\n" + "=" * 60)
    print("🔥 DIAGNOSTIC GPU NVIDIA POUR FER2013")
    print("=" * 60)
    
    # 1. Vérifier PyTorch
    torch = check_pytorch()
    if torch is None:
        sys.exit(1)
    
    # 2. Lister les GPUs
    num_gpus = check_gpus(torch)
    
    if num_gpus == 0:
        print("\n❌ Aucun GPU détecté!")
        sys.exit(1)
    
    # 3. Benchmark
    benchmark_gpu(torch, gpu_id=0)
    
    # 4. Recommandations
    recommend_settings(torch, num_gpus)
    
    print("\n" + "=" * 60)
    print("✅ DIAGNOSTIC TERMINÉ - GPU PRÊT!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
