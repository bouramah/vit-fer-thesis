# EmoVision — application web du DBFN

Application de démonstration du mémoire : reconnaissance des expressions faciales en direct (webcam), sur photo, et « défi des 7 émotions », avec le choix entre ResNet-50, ViT-Base, DBFN ou la comparaison des trois. Interface bilingue français / anglais.

## Fonctionnalités

- **Direct** : plusieurs visages suivis en temps réel, étiquette et confiance par visage, barres de probabilité.
- **Photo** : photos de groupe, explication visuelle Grad-CAM, retour « la prédiction est-elle juste ? ».
- **Défi des 7 émotions** : l'application demande chaque émotion et capture l'image dès que le DBFN la reconnaît ; récapitulatif par modèle.
- **Contributions consenties** : rien n'est conservé sans accord explicite (18 ans ou plus, personne photographiée ou accord de chacun, usage dans le mémoire). Un code permet le retrait à tout moment.
- **Administration** (`/admin`, mot de passe) : consultation, export zip, suppression.

## Architecture

| Élément | Rôle |
|---|---|
| `server/engine.py` | ResNet-50, ViT-Base et DBFN ; en mode « comparer », le ViT n'est calculé qu'une fois. Prétraitement identique à l'entraînement (niveaux de gris, 224×224, normalisation ImageNet). |
| `server/faces.py` | Détection YuNet (OpenCV) et recadrage calibré sur le cadrage de FER2013 (carré de 1,3 × le cadre détecté, légèrement remonté). |
| `server/main.py` | API FastAPI et service de l'interface. |
| `server/storage.py` | Contributions consenties et retours anonymes. |
| `web/` | Interface (HTML/JS, sans compilation). MediaPipe, chargé depuis un CDN, sert uniquement à animer les cadres entre deux analyses du serveur. |
| `tools/build_weights.py` | Assemble les trois modèles en un fichier de 257 Mo (ViT partagé, float16). |
| `tools/make_annex.py` | Produit les figures et l'exactitude par modèle pour l'annexe du mémoire, à partir des seules contributions consenties. |

Latence mesurée sur CPU (Apple M3 Max, 4 threads) : 22 ms par visage pour ResNet-50, 42 ms pour ViT-Base, 63 ms pour le DBFN, 120 ms pour les trois.

**Accélération ONNX Runtime** (`EMOVISION_BACKEND=onnx`, après `python app/tools/export_onnx.py`) : prédictions identiques à PyTorch (100 % d'accord sur 2 000 images de test), 8 à 22 % plus rapide selon le processeur. La variante `onnx-int8` (ViT quantifié) est déconseillée : elle fait perdre 0,7 point d'exactitude au DBFN et modifie 3 % des prédictions.

## Lancer en local

Depuis la racine du dépôt, après avoir placé les poids dans `app/weights/` (voir [weights/README.md](weights/README.md)) :

```bash
pip install -r app/requirements.txt
ADMIN_PASSWORD=choisir-un-mot-de-passe EMOVISION_THREADS=4 uvicorn app.server.main:app --port 8000
```

Puis ouvrir http://localhost:8000. La webcam n'est accessible qu'en `localhost` ou en HTTPS.

## Déployer sur un VPS (Docker + HTTPS automatique)

1. Faire pointer le nom de domaine (enregistrement A) vers l'adresse IP du VPS.
2. Installer Docker, cloner le dépôt, puis copier `app/deploy/.env.example` en `app/deploy/.env` et le remplir (`DOMAIN`, `ADMIN_PASSWORD`, `CONTACT_EMAIL`, `EMOVISION_THREADS`).
3. Depuis la racine du dépôt :

```bash
docker compose -f app/deploy/docker-compose.yml up -d --build
```

Caddy obtient le certificat HTTPS automatiquement. Les contributions sont conservées dans le volume Docker `emovision-data`.

## Déployer sur Hugging Face Spaces

Créer un Space de type Docker, y pousser le dépôt et ajouter, en tête du README du Space :

```yaml
---
title: EmoVision
sdk: docker
app_port: 8000
---
```

Le disque d'un Space gratuit est effacé à chaque redémarrage : les contributions doivent alors être exportées régulièrement depuis `/admin`.

## Générer l'annexe du mémoire

Après export depuis `/admin` (fichier zip à décompresser) :

```bash
python app/tools/make_annex.py --data chemin/vers/export --out annexe/
```
