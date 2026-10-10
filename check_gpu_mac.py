#!/usr/bin/env python3
"""
🍎 Vérification GPU pour Mac (MPS - Metal Performance Shaders)
===============================================================
Auteur: Ibrahima Douty OULARE

Ce script vérifie si ton Mac peut utiliser l'accélération GPU avec PyTorch.
- Mac Apple Silicon (M1/M2/M3/M4): Support MPS natif
- Mac Intel avec AMD GPU: Support MPS
- Mac Intel sans GPU dédié: CPU seulement
"""

import sys
import platform

def check_system():
    """Vérifier les informations système."""
    print("\n" + "=" * 60)
    print("🖥️  INFORMATIONS SYSTÈME")
    print("=" * 60)
    
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Version: {platform.mac_ver()[0]}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Processeur: {platform.processor()}")
    
    # Détecter Apple Silicon vs Intel
    if platform.machine() == "arm64":
        print(f"  Type: 🍎 Apple Silicon (M1/M2/M3/M4)")
        return "apple_silicon"
    else:
        print(f"  Type: Intel Mac")
        return "intel"


def check_pytorch():
    """Vérifier l'installation de PyTorch."""
    print("\n" + "=" * 60)
    print("🔥 VÉRIFICATION PYTORCH")
    print("=" * 60)
    
    try:
        import torch
        print(f"  ✅ PyTorch installé: {torch.__version__}")
        return torch
    except ImportError:
        print("  ❌ PyTorch n'est pas installé!")
        print("\n  Pour installer PyTorch sur Mac:")
        print("  pip install torch torchvision")
        return None


def check_mps(torch):
    """Vérifier le support MPS (Metal Performance Shaders)."""
    print("\n" + "=" * 60)
    print("⚡ VÉRIFICATION MPS (GPU Mac)")
    print("=" * 60)
    
    # Vérifier si MPS est disponible
    if not hasattr(torch.backends, 'mps'):
        print("  ❌ MPS non disponible (PyTorch trop ancien)")
        print("  Mise à jour recommandée: pip install --upgrade torch")
        return False
    
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    print(f"  MPS built: {'✅ Oui' if mps_built else '❌ Non'}")
    print(f"  MPS available: {'✅ Oui' if mps_available else '❌ Non'}")
    
    if not mps_available:
        print("\n  ⚠️ MPS non disponible. Causes possibles:")
        print("     - macOS < 12.3 (Monterey)")
        print("     - PyTorch sans support MPS")
        print("     - Hardware non compatible")
        return False
    
    return True


def test_mps_operations(torch):
    """Tester les opérations sur MPS."""
    print("\n" + "=" * 60)
    print("🧪 TEST DES OPÉRATIONS GPU (MPS)")
    print("=" * 60)
    
    try:
        device = torch.device("mps")
        print(f"  Device: {device}")
        
        # Test 1: Création de tenseur
        print("\n  Test 1: Création de tenseur sur GPU...")
        x = torch.randn(1000, 1000, device=device)
        print(f"    ✅ Tenseur créé: {x.shape} sur {x.device}")
        
        # Test 2: Opération matricielle
        print("\n  Test 2: Multiplication matricielle...")
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print(f"    ✅ Multiplication réussie: {z.shape}")
        
        # Test 3: Transfert CPU <-> GPU
        print("\n  Test 3: Transfert CPU <-> GPU...")
        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = cpu_tensor.to(device)
        back_to_cpu = gpu_tensor.cpu()
        print(f"    ✅ Transfert réussi")
        
        # Test 4: Réseau de neurones simple
        print("\n  Test 4: Réseau de neurones sur GPU...")
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to(device)
        
        input_data = torch.randn(32, 784, device=device)
        output = model(input_data)
        print(f"    ✅ Forward pass réussi: {output.shape}")
        
        # Test 5: Backward pass
        print("\n  Test 5: Backward pass (gradient)...")
        loss = output.sum()
        loss.backward()
        print(f"    ✅ Backward pass réussi")
        
        return True
        
    except Exception as e:
        print(f"\n  ❌ Erreur: {e}")
        return False


def benchmark_mps(torch):
    """Benchmark CPU vs MPS."""
    print("\n" + "=" * 60)
    print("📊 BENCHMARK CPU vs GPU (MPS)")
    print("=" * 60)
    
    import time
    import torch.nn as nn
    
    # Paramètres du benchmark
    batch_size = 64
    input_size = 224 * 224 * 3
    hidden_size = 1024
    num_iterations = 100
    
    # Modèle simple
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 7)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
    
    # Test CPU
    print("\n  🔵 Test CPU...")
    model_cpu = SimpleModel()
    x_cpu = torch.randn(batch_size, input_size)
    
    start = time.time()
    for _ in range(num_iterations):
        _ = model_cpu(x_cpu)
    cpu_time = time.time() - start
    print(f"    Temps: {cpu_time:.3f}s pour {num_iterations} itérations")
    print(f"    Vitesse: {num_iterations/cpu_time:.1f} it/s")
    
    # Test MPS
    if torch.backends.mps.is_available():
        print("\n  🟢 Test GPU (MPS)...")
        device = torch.device("mps")
        model_mps = SimpleModel().to(device)
        x_mps = torch.randn(batch_size, input_size, device=device)
        
        # Warmup
        for _ in range(10):
            _ = model_mps(x_mps)
        
        # Synchroniser avant le timing
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        
        start = time.time()
        for _ in range(num_iterations):
            _ = model_mps(x_mps)
        
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        
        mps_time = time.time() - start
        print(f"    Temps: {mps_time:.3f}s pour {num_iterations} itérations")
        print(f"    Vitesse: {num_iterations/mps_time:.1f} it/s")
        
        # Comparaison
        speedup = cpu_time / mps_time
        print(f"\n  📈 Speedup GPU vs CPU: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"  ✅ GPU est {speedup:.1f}x plus rapide!")
        else:
            print(f"  ⚠️ CPU plus rapide (normal pour petits modèles)")
    
    return True


def get_recommended_device(torch):
    """Retourner le device recommandé."""
    print("\n" + "=" * 60)
    print("🎯 RECOMMANDATION")
    print("=" * 60)
    
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"\n  ✅ Utilise: device = torch.device('mps')")
        print(f"\n  Dans train.py, lance avec:")
        print(f"     python train.py --model resnet18 --device mps")
    else:
        device = "cpu"
        print(f"\n  ⚠️ Utilise: device = torch.device('cpu')")
        print(f"\n  MPS non disponible, entraînement sur CPU")
    
    return device


def main():
    """Fonction principale."""
    print("\n" + "=" * 60)
    print("🍎 VÉRIFICATION GPU POUR MAC")
    print("=" * 60)
    
    # 1. Vérifier le système
    chip_type = check_system()
    
    # 2. Vérifier PyTorch
    torch = check_pytorch()
    if torch is None:
        print("\n❌ Installe d'abord PyTorch:")
        print("   pip install torch torchvision")
        sys.exit(1)
    
    # 3. Vérifier MPS
    mps_ok = check_mps(torch)
    
    if mps_ok:
        # 4. Tester les opérations
        test_ok = test_mps_operations(torch)
        
        if test_ok:
            # 5. Benchmark
            benchmark_mps(torch)
    
    # 6. Recommandation
    device = get_recommended_device(torch)
    
    print("\n" + "=" * 60)
    print("✅ VÉRIFICATION TERMINÉE")
    print("=" * 60)
    
    if mps_ok:
        print(f"""
🎉 Ton Mac peut utiliser l'accélération GPU!

Pour entraîner ResNet sur FER2013:

    python train.py --model resnet18 --device mps --batch_size 64
    python train.py --model resnet50 --device mps --batch_size 32

""")
    else:
        print(f"""
⚠️ GPU non disponible, utilisation du CPU.

Pour entraîner ResNet sur FER2013:

    python train.py --model resnet18 --device cpu --batch_size 32
    python train.py --model resnet50 --device cpu --batch_size 16

Note: L'entraînement sera plus lent sur CPU.
""")
    
    return device


if __name__ == "__main__":
    main()
