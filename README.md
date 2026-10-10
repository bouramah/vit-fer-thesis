# DBFN — Dual-Branch Fusion Network pour la reconnaissance d'expressions faciales

**Fusion ResNet-50 (CNN) + ViT-Base (Transformer) à backbones gelés pour la FER**

Code du mémoire de Master *« Apprentissage profond basé sur les Vision Transformers pour la reconnaissance automatique des expressions faciales »* (Master IASD, Université Kofi Annan de Guinée, 2025-2026) et de l'article associé :

> I. D. Oulare, A. Kermi, M. T. Laskri. *A Parameter-Efficient Dual-Branch Fusion of ResNet and Vision Transformer for Facial Expression Recognition.* CARI 2026 (accepté).

**Auteur :** Ibrahima Douty OULARE · **Encadrant :** Dr Adel KERMI · **Co-auteur :** Pr Mohamed Tayeb LASKRI

---

## Idée

Deux backbones **fine-tunés puis gelés** (ResNet-50 pour les indices locaux, ViT-Base pour le contexte global) ; seule une **tête de fusion de 1,58 M paramètres** (concaténation + MLP, 1,4 % du modèle) est entraînée. La fusion dépasse chaque backbone seul sur trois jeux de données, pour un coût de calcul égal à la somme des deux backbones.

| Dataset | ResNet-50 | ViT-Base | **DBFN** (moyenne ± écart-type, 3 graines) | **DBFN** (meilleur run) |
|---|---:|---:|---:|---:|
| FER2013 | 70,35 | 72,64 | 73,51 ± 0,11 | **73,86** (74,17 avec TTA 10-crop) |
| RAF-DB | 84,39 | 86,15 | 86,92 ± 0,69 | **87,71** |
| AffectNet-7 | 56,86 | 57,00 | 59,98 ± 0,57 | **60,63** |

Exactitude (%) sur les jeux de test officiels. Détail complet (macro-F1, ablations, complexité, complémentarité) : [RESULTS.md](RESULTS.md).

👉 **Pour installer, évaluer, ré-entraîner ou lancer la démo : voir [GUIDE.md](GUIDE.md).**

---

## Structure

```
├── models/
│   ├── resnet.py              # ResNet-18/50 pour la FER
│   ├── vit.py                 # ViT-Small/Base, DeiT-Small (timm)
│   ├── hybrid.py              # DBFN : branches + fusions (concat / gating / cross-attention)
│   └── hybrid_tokens.py       # variante : cross-attention au niveau des tokens
├── utils/                     # jeu de données (format dossiers), métriques, pertes
├── train_v2.py                # entraînement des backbones ResNet
├── train_vit_v3.py            # entraînement des backbones ViT / DeiT
├── train_hybrid.py            # entraînement de la tête de fusion DBFN
├── train_hybrid_tokens.py     # entraînement de la variante tokens
├── evaluate_hybrid_tta.py     # évaluation du DBFN (+ TTA none|hflip|5crop|10crop)
├── benchmark_complexity.py    # paramètres, GFLOPs, taille, latence, FPS
├── complementarity_dbfn.py    # analyse d'accord entre branches + Grad-CAM
├── attention_compare.py       # Grad-CAM (CNN) vs attention (ViT)
├── build_affectnet7.py        # construction d'AffectNet-7 (contempt exclu)
├── download_dataset.py        # téléchargement de FER2013 (Kaggle)
├── recreate_data_symlinks.sh  # mise au format dossiers de RAF-DB
├── demo.py                    # démo image / webcam (y compris DBFN complet)
├── best_models.sh             # commandes exactes des backbones retenus
├── RESULTS.md                 # tous les résultats consolidés
└── GUIDE.md                   # guide pas à pas
```

Les jeux de données et les checkpoints ne sont pas versionnés (licences et taille). Les checkpoints principaux sont fournis dans la [Release v1.0](https://github.com/bouramah/vit-fer-thesis/releases).

## Références principales

- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- A. Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021.
- I. Goodfellow et al., FER2013, ICONIP 2013 · S. Li et al., RAF-DB, CVPR 2017 · A. Mollahosseini et al., AffectNet, IEEE TAC 2019.
