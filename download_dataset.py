"""
Script de téléchargement et préparation de FER2013
===================================================
Auteur: Ibrahima Douty OULARE

Ce script télécharge FER2013 depuis Kaggle et prépare la structure.

Prérequis:
    1. Avoir un compte Kaggle
    2. Télécharger votre API key depuis https://www.kaggle.com/account
    3. Placer kaggle.json dans ~/.kaggle/ (Linux/Mac) ou %USERPROFILE%\.kaggle\ (Windows)
    4. pip install kaggle
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path


def check_kaggle_api():
    """Vérifier si l'API Kaggle est configurée."""
    try:
        import kaggle
        return True
    except:
        return False


def download_fer2013(output_dir: str = "data"):
    """
    Télécharger FER2013 depuis Kaggle.
    
    Args:
        output_dir: Dossier de destination
    """
    output_path = Path(output_dir)
    fer2013_path = output_path / "fer2013"
    
    # Vérifier si déjà téléchargé
    if fer2013_path.exists() and (fer2013_path / "train").exists():
        print(f"✓ FER2013 déjà présent dans {fer2013_path}")
        return True
    
    # Vérifier l'API Kaggle
    if not check_kaggle_api():
        print("❌ Erreur: L'API Kaggle n'est pas configurée.")
        print("\nInstructions:")
        print("1. pip install kaggle")
        print("2. Aller sur https://www.kaggle.com/account")
        print("3. Cliquer sur 'Create New API Token'")
        print("4. Placer kaggle.json dans ~/.kaggle/")
        print("5. chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("📥 Téléchargement de FER2013 depuis Kaggle...")
        
        # Initialiser l'API
        api = KaggleApi()
        api.authenticate()
        
        # Créer le dossier
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Télécharger
        api.dataset_download_files(
            "msambare/fer2013",
            path=str(output_path),
            unzip=True
        )
        
        print(f"✓ Téléchargement terminé dans {output_path}")
        
        # Vérifier la structure
        if (fer2013_path / "train").exists():
            print("✓ Structure vérifiée")
            return True
        else:
            # Peut-être que c'est extrait directement dans output_path
            if (output_path / "train").exists():
                # Déplacer dans fer2013/
                fer2013_path.mkdir(exist_ok=True)
                shutil.move(str(output_path / "train"), str(fer2013_path / "train"))
                shutil.move(str(output_path / "test"), str(fer2013_path / "test"))
                print("✓ Structure réorganisée")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return False


def verify_dataset(data_dir: str = "data/fer2013"):
    """
    Vérifier la structure du dataset.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Dossier non trouvé: {data_path}")
        return False
    
    expected_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    print("\n📊 Vérification du dataset FER2013...")
    print("-" * 50)
    
    total_train = 0
    total_test = 0
    
    for split in ['train', 'test']:
        split_path = data_path / split
        
        if not split_path.exists():
            print(f"❌ Dossier manquant: {split_path}")
            return False
        
        print(f"\n{split.upper()}:")
        
        for class_name in expected_classes:
            class_path = split_path / class_name
            
            if not class_path.exists():
                print(f"  ❌ {class_name}: MANQUANT")
                continue
            
            # Compter les images
            count = len(list(class_path.glob("*.jpg"))) + len(list(class_path.glob("*.png")))
            print(f"  ✓ {class_name:<10}: {count:>5} images")
            
            if split == 'train':
                total_train += count
            else:
                total_test += count
    
    print("-" * 50)
    print(f"Total Train: {total_train} images")
    print(f"Total Test:  {total_test} images")
    print(f"Total:       {total_train + total_test} images")
    print("-" * 50)
    
    # Vérifier les totaux attendus
    if total_train > 25000 and total_test > 5000:
        print("✓ Dataset vérifié avec succès!")
        return True
    else:
        print("⚠️ Attention: Le nombre d'images semble incomplet")
        return False


def download_manual_instructions():
    """
    Afficher les instructions pour le téléchargement manuel.
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           TÉLÉCHARGEMENT MANUEL DE FER2013                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Aller sur:                                                   ║
║     https://www.kaggle.com/datasets/msambare/fer2013             ║
║                                                                  ║
║  2. Cliquer sur "Download" (nécessite un compte Kaggle)          ║
║                                                                  ║
║  3. Extraire le ZIP téléchargé                                   ║
║                                                                  ║
║  4. Organiser la structure comme suit:                           ║
║                                                                  ║
║     data/                                                        ║
║     └── fer2013/                                                 ║
║         ├── train/                                               ║
║         │   ├── angry/                                           ║
║         │   ├── disgust/                                         ║
║         │   ├── fear/                                            ║
║         │   ├── happy/                                           ║
║         │   ├── neutral/                                         ║
║         │   ├── sad/                                             ║
║         │   └── surprise/                                        ║
║         └── test/                                                ║
║             └── (même structure)                                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎭 SETUP DATASET FER2013")
    print("=" * 60)
    
    # Essayer le téléchargement automatique
    success = download_fer2013("data")
    
    if not success:
        print("\n⚠️ Téléchargement automatique impossible.")
        download_manual_instructions()
    
    # Vérifier dans tous les cas
    verify_dataset("data/fer2013")
