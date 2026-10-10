# Guide de lancement du projet DBFN

Ce guide explique comment installer le projet, reproduire le résultat principal du mémoire (73,86 % sur FER2013) à partir du checkpoint publié, ré-entraîner les modèles et lancer la démo. Toutes les commandes se lancent depuis la racine du dépôt.

---

## 1. Installation

Python 3.9 à 3.12 recommandé.

```bash
git clone https://github.com/bouramah/vit-fer-thesis.git
cd vit-fer-thesis
python -m venv venv_fer
source venv_fer/bin/activate        # Windows : venv_fer\Scripts\activate
pip install -r requirements.txt
```

Pour un GPU NVIDIA, installer la version de PyTorch adaptée à votre version de CUDA (voir pytorch.org) avant `requirements.txt`. Les résultats du mémoire ont été obtenus avec PyTorch 2.5.1 (CUDA 12.1) et timm 1.0.24 sur NVIDIA A100.

## 2. Jeux de données

Les jeux de données ne sont pas fournis (licences). Tous les scripts attendent le même format :

```
<data_dir>/train/<classe>/*.jpg
<data_dir>/test/<classe>/*.jpg
```

avec les 7 classes `angry, disgust, fear, happy, neutral, sad, surprise`.

| Jeu de données | Dossier attendu | Préparation |
|---|---|---|
| FER2013 | `data/fer2013` | `python download_dataset.py` (clé API Kaggle requise, voir l'en-tête du script) |
| RAF-DB (aligné) | `data_rafdb_fer` | placer l'archive officielle extraite dans `data_rafdb/DATASET/{train,test}/{1..7}/`, puis `bash recreate_data_symlinks.sh` |
| AffectNet-7 | `data_affectnet_fer` | extraire l'archive officielle dans `data_affectnet/` et les annotations `.pt` dans `/tmp/affnet_pt/`, puis `python build_affectnet7.py` (la classe *contempt* est exclue) |

RAF-DB et AffectNet sont distribués par leurs auteurs sur demande, pour un usage de recherche non commercial.

## 3. Reproduire le résultat principal (sans ré-entraîner)

Télécharger depuis la [Release v1.0](https://github.com/bouramah/vit-fer-thesis/releases) le checkpoint du DBFN de référence et le placer ainsi :

```
checkpoints_hybrid/dbfn_concat_frozen_20260227_173625_epoch1_best.pth
```

Ce fichier (436 Mo) contient les poids complets : les deux backbones et la tête de fusion. Puis :

```bash
python evaluate_hybrid_tta.py --data_dir data/fer2013 --fusion concat --tta none \
  --checkpoint checkpoints_hybrid/dbfn_concat_frozen_20260227_173625_epoch1_best.pth \
  --num_workers 0
```

Résultat attendu : **Accuracy 73,86 %, Macro-F1 72,71 %** (vérifié, environ 2 minutes sur Apple M3 Max avec `--device mps`). Avec `--tta 10crop` : 74,17 % / 73,02 %.

> **macOS / Windows :** utiliser `--num_workers 0`. Avec plusieurs workers, le script s'interrompt (erreur `Can't get local object 'main.<locals>.collate_pil'`), car ces systèmes lancent les workers en mode *spawn*. Sous Linux, la valeur par défaut fonctionne.

## 4. Ré-entraîner

L'entraînement se fait en deux étapes, pour chaque jeu de données.

**Étape 1 — Backbones.** Les commandes exactes des runs retenus (FER2013) figurent dans [best_models.sh](best_models.sh). Pour un autre jeu de données, changer `--data_dir` :

```bash
python train_v2.py --model resnet50 --data_dir data/fer2013 --batch_size 64 --epochs 80 --two_stage \
  --lr 1e-4 --mixup 0.1 --label_smoothing 0.1 --dropout 0.3 --checkpoint_dir checkpoints_resnet50

python train_vit_v3.py --model vit_base --data_dir data/fer2013 --epochs 90 --batch_size 64 --lr 1e-4 \
  --weight_decay 0.05 --drop_path 0.18 --dropout 0.20 --mixup 0.10 --label_smoothing 0.05 --seed 42 \
  --checkpoint_dir checkpoints_vit_base
```

**Étape 2 — Tête de fusion, backbones gelés.**

```bash
python train_hybrid.py --data_dir data/fer2013 --fusion concat \
  --freeze_backbones --unfreeze_epoch 999 --split_seed 42 --seed 42 \
  --cnn_checkpoint checkpoints_resnet50/<meilleur>.pth \
  --vit_checkpoint checkpoints_vit_base/<meilleur>.pth \
  --checkpoint_dir checkpoints_fusion
```

Trois options sont indispensables pour reproduire le protocole du mémoire :

- `--freeze_backbones --unfreeze_epoch 999` : les backbones restent gelés pendant tout l'entraînement. Sans `--unfreeze_epoch 999`, la valeur par défaut (10) dégèle partiellement les backbones à la 10ᵉ époque.
- `--split_seed 42` : la séparation entraînement/validation de la tête doit être celle des backbones. Sinon, la validation contient des images déjà vues par les backbones (fuite de données).
- `--seed` : faire varier ce paramètre (42, 123, 2024) pour l'étude multi-graines, en gardant `--split_seed 42`.

`--fusion gating` et `--fusion cross_attention` reproduisent l'ablation des fusions. La variante au niveau des tokens s'entraîne avec `train_hybrid_tokens.py`.

Durée indicative sur une A100 : environ 20 minutes pour la tête de fusion, plusieurs heures pour chaque backbone.

## 5. Analyses

```bash
# Complexité : paramètres, GFLOPs, taille, latence, débit (aucun checkpoint requis)
python benchmark_complexity.py --device cuda

# Complémentarité entre branches + Grad-CAM
python complementarity_dbfn.py --data_dir data/fer2013 \
  --resnet_ckpt <resnet50>.pth --vit_ckpt <vit_base>.pth --dbfn_ckpt <dbfn>.pth --out_dir results_interpretability

# Grad-CAM (CNN) vs attention (ViT) sur FER2013 : chemins des checkpoints fixés en tête du script
# (checkpoints_run3_resnet50/... et checkpoints_vit_base_v3_run1/..., fournis dans la Release)
python attention_compare.py
```

## 6. Démo (image ou webcam)

```bash
# Image : enregistre le résultat annoté
python demo.py --image photo.jpg --model dbfn \
  --checkpoint checkpoints_hybrid/dbfn_concat_frozen_20260227_173625_epoch1_best.pth --output resultat.jpg

# Webcam en temps réel (touche q pour quitter)
python demo.py --webcam --model dbfn \
  --checkpoint checkpoints_hybrid/dbfn_concat_frozen_20260227_173625_epoch1_best.pth
```

La démo fonctionne hors ligne et sur CPU (environ 70 ms par visage sur Apple M3 Max). Le visage est détecté par une cascade de Haar d'OpenCV.

> **OpenCV ≥ 5 :** les fichiers de cascade de Haar ne sont plus inclus, et la démo analyse alors l'image entière sans le signaler clairement (message « Cascade classifier non trouvé »), ce qui fausse les prédictions. `requirements.txt` impose donc `opencv-python<5`.

`--model` accepte aussi `resnet18`, `resnet50`, `vit_small`, `vit_base` et `deit_small`, avec le checkpoint correspondant.

## 7. Correspondance résultats / checkpoints

Le fichier [RESULTS.md](RESULTS.md) (section 8) associe chaque résultat du mémoire au checkpoint qui l'a produit.
