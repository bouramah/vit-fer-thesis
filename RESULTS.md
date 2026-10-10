# RESULTS — Récapitulatif complet (mémoire)

Projet **DBFN** (Dual-Branch Fusion Network : ResNet-50 + ViT-Base, fusion gelée) pour la reconnaissance d'expressions faciales.
Tous les résultats sur les **test sets officiels**. Recette identique sur les 3 datasets (backbones fine-tunés depuis ImageNet → gelés → tête de fusion entraînée ; pipeline 224×224, normalisation ImageNet ; sélection sur macro-F1 val ; `split_seed=42` pour la tête).

Matériel : NVIDIA A100-40GB · PyTorch 2.5.1+cu121 · timm 1.0.24.

---

## 1. FER2013

> **Jeu de test FER2013 :** 7 178 images = réunion des sous-ensembles **PublicTest** et **PrivateTest** (3 589 images chacun) de la compétition d'origine. Sur le seul PrivateTest, souvent utilisé dans la littérature, le DBFN de référence atteint **74,28 %** (macro-F1 73,24 %), contre 73,45 % sur le PublicTest.

### 1.1 Modèles individuels (test set, 7 178 images)

| Modèle | Type | Params | Accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|
| ResNet-18 | CNN | 11.2M | 70.23 | 68.99 |
| ResNet-50 | CNN | 23.5M | 70.35 | 69.78 |
| DeiT-Small | ViT | 22M | 68.97 | 67.82 |
| ViT-Small | ViT | 22M | 70.56 | 69.67 |
| **ViT-Base** | ViT | 86M | **72.64** | **71.24** |

### 1.2 Ablation des fusions DBFN (backbones gelés, moyenne ± std sur 3 seeds)

| Fusion | Params entraîn. | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| **Concat** | 1.58M | 73.51 ± 0.11 | 72.20 ± 0.23 |
| Gating | 2.10M | 73.31 ± 0.31 | 72.27 ± 0.25 |
| Cross-Attention | 4.73M | 73.32 ± 0.18 | 72.19 ± 0.16 |

*Meilleur run concat (chiffre titre) : **73.86 % / 72.71 %**. Par seed (Acc/F1) : concat 73.63/72.47 · 73.49/72.11 · 73.42/72.03.*
Fusion gelée (concat 73.86) > fine-tuning end-to-end (cross-attn FT 72.95) malgré 71× moins de paramètres entraînables.

### 1.3 Test-Time Augmentation (DBFN concat)

| TTA | Passes | Accuracy (%) | Macro-F1 (%) | Latence (ms/img) | FPS |
|---|---:|---:|---:|---:|---:|
| None | 1 | 73.86 | 72.71 | 6.6 | 152 |
| H-Flip | 2 | 73.77 | 72.61 | 8.7 | 115 |
| 5-Crop | 5 | 74.00 | 72.93 | 16.9 | 59 |
| **10-Crop** | 10 | **74.17** | **73.02** | 31.4 | 32 |

### 1.4 Métriques par classe (DBFN concat, no-TTA, 73.86 %)

| Classe | Précision (%) | Rappel (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| Angry | 62.91 | 70.46 | 66.47 | 958 |
| Disgust | 86.90 | 65.77 | 74.87 | 111 |
| Fear | 67.25 | 52.54 | 58.99 | 1024 |
| Happy | 89.99 | 90.19 | 90.09 | 1774 |
| Neutral | 69.21 | 75.10 | 72.03 | 1233 |
| Sad | 62.74 | 63.19 | 62.96 | 1247 |
| Surprise | 82.69 | 84.48 | 83.57 | 831 |
| **Weighted** | 73.93 | 73.86 | — | 7178 |
| **Macro** | — | — | **72.71** | — |

### 1.5 Ablation régularisation (ViT-Small, incrémentale)

| Configuration | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Baseline (reg. minimale) | 69.09 | 66.11 |
| + Weight decay 0.05 | 69.74 | 67.10 |
| + Drop path 0.1 | 69.74 | 67.10 |
| + Weighted sampler | 70.26 | 69.04 |
| + Sans mixup (best) | 70.56 | 69.67 |

---

## 2. RAF-DB (test 3 068 images)

| Modèle | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| ResNet-50 | 84.39 | 77.36 |
| ViT-Base | 86.15 | 78.78 |
| **DBFN (concat, gelé)** | **86.92 ± 0.69** (best 87.71) | **80.36 ± 0.71** |

*Par seed (Acc/F1) : 87.71/81.18 · 86.57/80.02 · 86.47/79.88.*

---

## 3. AffectNet-7 (test équilibré 3 500 images, 500/classe ; train 283 901, contempt exclu)

| Modèle | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| ResNet-50 | 56.86 | 55.45 |
| ViT-Base | 57.00 | 56.02 |
| **DBFN (concat, gelé)** | **59.98 ± 0.57** (best 60.63) | **59.25 ± 0.58** |

*Par seed (Acc/F1) : 59.71/58.80 · 60.63/59.90 · 59.60/59.05.*
*Note : recette identique sans weighted sampler (déséquilibre 35×) ; SOTA AffectNet-7 ≈ 63-67 %.*

---

## 4. Complexité & efficacité (A100, batch 1)

| Modèle | Params total | Params entraîn. | GFLOPs | Taille (Mo) | Latence (ms) | FPS (bs64) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 23.5M | 23.5M | 8.2 | 89.9 | 3.6 | 2780 |
| ViT-Base | 85.8M | 85.8M | 35.1 | 327.3 | 4.6 | 436 |
| **DBFN-concat** | 110.9M | **1.58M** | 43.3 | 423.2 | 6.6 | 381 |
| DBFN-cross-attn | 114.0M | 4.73M | 43.3 | 435.2 | 7.3 | 378 |

*GFLOPs additifs : 43.3 = 8.2 (ResNet) + 35.1 (ViT) → fusion négligeable. Seuls 1,4 % des params sont entraînés (« parameter-efficient »).*

---

## 5. Analyse de complémentarité (par dataset)

Prédictions par échantillon des 3 modèles sur le test set.

| Dataset | Les 2 ✓ | ResNet seul ✓ | ViT seul ✓ | Les 2 ✗ | XOR (litigieux) | Litigieux : ResNet / ViT / **DBFN** |
|---|---:|---:|---:|---:|---:|---|
| FER2013 | 63.6 | 6.8 | 9.1 | 20.6 | 15.9 % (N=1138) | 42.8 / 57.2 / **62.7** |
| RAF-DB | 78.9 | 5.5 | 7.2 | 8.4 | 12.7 % (N=390) | 43.1 / 56.9 / **65.9** |
| AffectNet-7 | 50.9 | 6.0 | 6.1 | 37.0 | 12.1 % (N=425) | 49.4 / 50.6 / **69.4** |

Sur le sous-ensemble litigieux, le DBFN dépasse la meilleure branche seule de **+5.5 / +9.0 / +18.8 points** → complémentarité **décisionnelle** (pas spatiale, cf. §6).

---

## 6. ★ Comparaison inter-datasets (résultat central)

| Dataset | ResNet-50 | ViT-Base | DBFN (best) | Gain vs ViT | Gain complémentarité (litigieux) |
|---|---:|---:|---:|---:|---:|
| FER2013 | 70.35 | 72.64 | 73.86 (74.17 +TTA) | +1.22 | +5.5 |
| RAF-DB | 84.39 | 86.15 | 87.71 | +1.56 | +9.0 |
| AffectNet-7 | 56.86 | 57.00 | 60.63 | +3.63 | +18.8 |

**Conclusions robustes (3 datasets) :**
1. DBFN > les deux backbones **systématiquement** → la recette généralise.
2. Le **gain de fusion croît avec la difficulté** du dataset (+1.22 → +1.56 → +3.63).
3. La **complémentarité croît** aussi (+5.5 → +9.0 → +18.8 pts) : plus les branches sont faillibles seules, plus la fusion apporte.
4. La complémentarité est **décisionnelle, pas spatiale** : les cartes Grad-CAM (CNN) et l'attention ViT sont spatialement décorrélées (corr ≈ 0) — l'attention brute du ViT est dominée par des artefacts (« attention sinks »). La preuve de complémentarité est quantitative (§5), pas visuelle.

---

## 7. Figures (results_interpretability/)

- `complementarity_dbfn.png` — complémentarité FER2013 (a: accord / b: litigieux / c: Grad-CAM).
- `complementarity_dbfn_rafdb.png` — idem RAF-DB.
- `complementarity_dbfn_affectnet.png` — idem AffectNet-7.
- `attention_compare.png` — Grad-CAM ResNet vs attention ViT (justifie le §6.4).

## 8. Checkpoints (reproductibilité)

| Rôle | Chemin |
|---|---|
| ResNet-50 FER2013 | `checkpoints_run3_resnet50/resnet50_v2_epoch61_best.pth` |
| ViT-Base FER2013 | `checkpoints_vit_base_v3_run1/vit_base_v3_epoch86_best.pth` |
| **DBFN FER2013 (titre 73.86)** | `checkpoints_hybrid/dbfn_concat_frozen_20260227_173625_epoch1_best.pth` |
| ResNet-18 / ViT-Small / DeiT-Small | `checkpoints_run2/` · `checkpoints_vit_small_v3_sampler_nomixup/` · `checkpoints_deit_small_v3/` |
| Multi-seed / ablation FER2013 | `checkpoints_{ablation,seed123,seed2024}_{concat,gating,cross_attention}/` |
| RAF-DB | `checkpoints_rafdb_{resnet50,vit_base,fusion_seed42/123/2024}/` |
| AffectNet-7 | `checkpoints_affectnet_{resnet50,vit_base,fusion_seed42/123/2024}/` |

Données chiffrées brutes : `results_complexity/complexity_metrics.json` · `results_interpretability/*.npz`. Les checkpoints principaux sont fournis dans la Release v1.0 du dépôt.
