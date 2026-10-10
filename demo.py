#!/usr/bin/env python3
"""
🎭 Démonstration - Test de Reconnaissance d'Expressions Faciales
=================================================================
Auteur: Ibrahima Douty OULARE
Master IASD - UKAG

Supporte maintenant:
✅ ResNet18 / ResNet50
✅ ViT-Small / ViT-Base / DeiT-Small (timm via ViTForFER)

Usage:
    # ViT-Base meilleur modèle (exemple)
    python demo.py --image photo.jpg --model vit_base --checkpoint checkpoints_vit_base_v3_run1/vit_base_v3_epoch86_best.pth

    # Webcam + ViT-Base
    python demo.py --webcam --model vit_base --checkpoint checkpoints_vit_base_v3_run1/vit_base_v3_epoch86_best.pth
"""

import argparse
import os
import sys
import time
import platform
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import modèles
from models.resnet import create_resnet18, create_resnet50
from models.vit import ViTForFER


# ==================== CONFIGURATION ====================

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTIONS_FR = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Neutre', 'Tristesse', 'Surprise']
EMOTIONS_EMOJI = ['😠', '🤢', '😨', '😊', '😐', '😢', '😲']

EMOTION_COLORS = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 128, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 255),
    'Neutral': (128, 128, 128),
    'Sad': (255, 0, 0),
    'Surprise': (0, 165, 255)
}


# ==================== SYSTEM INFO ====================

def get_os_pretty():
    try:
        import distro  # type: ignore
        name = distro.name(pretty=True)
        if name:
            return name
    except Exception:
        pass
    return platform.platform()

def print_system_info(device: torch.device):
    print("\n" + "=" * 70)
    print("🧾 CONFIGURATION ENVIRONNEMENT / MATERIEL")
    print("=" * 70)
    print(f"OS: {get_os_pretty()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device demandé: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(idx)
            vram_gb = prop.total_memory / (1024 ** 3)
            print(f"CUDA version (torch): {torch.version.cuda}")
            print(f"cuDNN: {torch.backends.cudnn.version()}")
            print(f"GPU: {prop.name}")
            print(f"VRAM: {vram_gb:.1f} GB")
        except Exception as e:
            print(f"⚠️ Impossible de lire les infos GPU: {e}")
    print("=" * 70 + "\n")


# ==================== PREPROCESSING ====================

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ==================== MODEL LOADING ====================

def load_model(model_name: str, checkpoint_path: str, device: torch.device):
    print(f"\n[MODEL] Chargement de {model_name}...")

    # --- Créer le modèle ---
    if model_name == "resnet18":
        model = create_resnet18(num_classes=7, pretrained=False)

    elif model_name == "resnet50":
        model = create_resnet50(num_classes=7, pretrained=False)

    elif model_name in ("vit_small", "vit_base", "deit_small"):
        timm_map = {
            "vit_small": "vit_small_patch16_224",
            "vit_base": "vit_base_patch16_224",
            "deit_small": "deit_small_patch16_224",
        }
        model = ViTForFER(
            model_name=timm_map[model_name],
            num_classes=7,
            pretrained=False,   # IMPORTANT: on charge NOS poids, pas ImageNet ici
            dropout=0.0,        # dropout inutile en eval
            drop_path_rate=0.0  # drop_path inutile en eval
        )

    elif model_name == "dbfn":
        # DBFN complet (ResNet-50 + ViT-Base + tête de fusion). Le checkpoint DBFN
        # contient déjà les poids des deux branches -> pas besoin de checkpoints séparés.
        from models.hybrid import create_dbfn
        model = create_dbfn(
            cnn_checkpoint=None, vit_checkpoint=None,
            fusion_type="concat", freeze_backbones=False,
            proj_dim=512, dropout=0.0, drop_path_rate=0.0, num_classes=7,
            imagenet_fallback=False  # pas de téléchargement ImageNet : le checkpoint DBFN fournit tous les poids
        )

    else:
        raise ValueError(f"Modèle non supporté: {model_name}")

    # --- Charger checkpoint ---
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint non trouvé: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Compat: certains checkpoints peuvent contenir directement state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    # strict=False pour le DBFN (tolère d'éventuels buffers de drop_path)
    missing, unexpected = model.load_state_dict(state, strict=(model_name != "dbfn"))
    if model_name == "dbfn":
        print(f"  [DBFN] chargé (missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device)
    model.eval()

    val_f1 = checkpoint.get('val_f1', None) if isinstance(checkpoint, dict) else None
    epoch = checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'

    if val_f1 is None:
        print(f"✅ Modèle chargé (epoch {epoch}, Val F1: N/A)")
    else:
        try:
            print(f"✅ Modèle chargé (epoch {epoch}, Val F1: {float(val_f1):.2f}%)")
        except Exception:
            print(f"✅ Modèle chargé (epoch {epoch}, Val F1: {val_f1})")

    return model


# ==================== FACE DETECTION ====================

def load_face_detector():
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]
    for path in cascade_paths:
        if os.path.exists(path):
            return cv2.CascadeClassifier(path)
    print("⚠️ Cascade classifier non trouvé, utilisation sans détection de visage")
    return None

def detect_faces(image, face_cascade):
    if face_cascade is None:
        h, w = image.shape[:2]
        return [(0, 0, w, h)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )
    return faces


# ==================== TIMING HELPERS ====================

def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def measure_inference_ms(model, input_tensor, device: torch.device, runs: int = 1):
    warmup = min(10, max(1, runs // 5))
    for _ in range(warmup):
        _ = model(input_tensor)
    _sync_if_cuda(device)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(input_tensor)
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


# ==================== PREDICTION ====================

@torch.no_grad()
def predict_emotion(model, face_img, transform, device, benchmark_runs: int = 1):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    input_tensor = transform(face_pil).unsqueeze(0).to(device)

    infer_ms_mean, infer_ms_std = measure_inference_ms(model, input_tensor, device, runs=benchmark_runs)

    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]

    confidence, predicted_idx = torch.max(probabilities, 0)

    return (
        predicted_idx.item(),
        EMOTIONS[predicted_idx.item()],
        confidence.item() * 100,
        probabilities.cpu().numpy() * 100,
        infer_ms_mean,
        infer_ms_std
    )


# ==================== VISUALIZATION ====================

def draw_prediction(image, face_rect, emotion, confidence, probs, infer_ms=None):
    x, y, w, h = face_rect
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    if infer_ms is None:
        text = f"{emotion}: {confidence:.1f}%"
    else:
        text = f"{emotion}: {confidence:.1f}% | {infer_ms:.1f} ms"

    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x, y - 30), (x + text_w + 10, y), color, -1)
    cv2.putText(image, text, (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    bar_x = image.shape[1] - 200
    bar_y_start = 30
    bar_height = 25
    bar_max_width = 150

    cv2.putText(image, "Probabilites:", (bar_x, bar_y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (emo, prob) in enumerate(zip(EMOTIONS, probs)):
        y_pos = bar_y_start + i * (bar_height + 5)
        bar_width = int(prob / 100 * bar_max_width)

        cv2.rectangle(image, (bar_x, y_pos), (bar_x + bar_max_width, y_pos + bar_height),
                      (50, 50, 50), -1)

        bar_color = EMOTION_COLORS.get(emo, (200, 200, 200))
        cv2.rectangle(image, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height),
                      bar_color, -1)

        cv2.putText(image, f"{emo[:3]}: {prob:.0f}%", (bar_x + 5, y_pos + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image


def print_prediction(emotion_idx, emotion, confidence, probs, infer_ms_mean=None, infer_ms_std=None):
    print("\n" + "=" * 60)
    print(f"🎭 ÉMOTION DÉTECTÉE: {EMOTIONS_EMOJI[emotion_idx]} {emotion.upper()}")
    print(f"   Confiance: {confidence:.1f}%")
    if infer_ms_mean is not None:
        if infer_ms_std is None:
            print(f"   Temps prédiction (1 visage): {infer_ms_mean:.2f} ms")
        else:
            print(f"   Temps prédiction (1 visage): {infer_ms_mean:.2f} ms (± {infer_ms_std:.2f})")
    print("=" * 60)
    print("\n📊 Probabilités:")

    sorted_idx = np.argsort(probs)[::-1]
    for idx in sorted_idx:
        bar_len = int(probs[idx] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        emoji = EMOTIONS_EMOJI[idx]
        print(f"   {emoji} {EMOTIONS[idx]:<10} {bar} {probs[idx]:5.1f}%")
    print()


# ==================== MAIN MODES ====================

def test_image(image_path: str, model, transform, device, face_cascade, output_path=None, benchmark_runs: int = 1):
    print(f"\n[IMAGE] Chargement de {image_path}...")

    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Impossible de lire l'image: {image_path}")
        return

    faces = detect_faces(image, face_cascade)

    if len(faces) == 0:
        print("⚠️ Aucun visage détecté, analyse de l'image entière...")
        h, w = image.shape[:2]
        faces = [(0, 0, w, h)]
    else:
        print(f"✅ {len(faces)} visage(s) détecté(s)")

    infer_times = []

    for i, (x, y, w, h) in enumerate(faces):
        face_img = image[y:y + h, x:x + w]

        emotion_idx, emotion, confidence, probs, infer_ms_mean, infer_ms_std = predict_emotion(
            model, face_img, transform, device, benchmark_runs=benchmark_runs
        )

        infer_times.append(infer_ms_mean)

        if len(faces) > 1:
            print(f"\n--- Visage {i + 1} ---")
        print_prediction(emotion_idx, emotion, confidence, probs, infer_ms_mean, infer_ms_std)

        image = draw_prediction(image, (x, y, w, h), emotion, confidence, probs, infer_ms=infer_ms_mean)

    if infer_times:
        print(f"⏱️ Temps moyen (sur {len(infer_times)} visage(s)) : {float(np.mean(infer_times)):.2f} ms/visage")

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"✅ Résultat sauvegardé: {output_path}")
    else:
        cv2.imshow('Emotion Detection', image)
        print("Appuie sur une touche pour fermer...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_webcam(model, transform, device, face_cascade, benchmark_runs: int = 1):
    print("\n[WEBCAM] Démarrage de la webcam...")
    print("Appuie sur 'q' pour quitter, 's' pour sauvegarder une capture")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    faces = []
    t_hist = deque(maxlen=30)
    infer_hist = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        t0_frame = time.perf_counter()

        if frame_count % 3 == 0:
            faces = detect_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            if face_img.size == 0:
                continue

            emotion_idx, emotion, confidence, probs, infer_ms_mean, infer_ms_std = predict_emotion(
                model, face_img, transform, device, benchmark_runs=benchmark_runs
            )
            infer_hist.append(infer_ms_mean)
            frame = draw_prediction(frame, (x, y, w, h), emotion, confidence, probs, infer_ms=infer_ms_mean)

        t1_frame = time.perf_counter()
        t_hist.append(t1_frame - t0_frame)

        fps = 1.0 / (sum(t_hist) / len(t_hist)) if len(t_hist) >= 2 else 0.0
        avg_infer = float(np.mean(infer_hist)) if infer_hist else None

        info_line = f"FPS: {fps:.1f}"
        if avg_infer is not None:
            info_line += f" | Inference: {avg_infer:.1f} ms/visage"

        cv2.putText(frame, info_line, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "Q: Quitter | S: Sauvegarder", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Emotion Detection - Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Capture sauvegardée: {filename}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam fermée")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description="🎭 Démonstration de Reconnaissance d'Expressions Faciales",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image', type=str, help='Chemin vers une image')
    mode.add_argument('--webcam', action='store_true', help='Utiliser la webcam')

    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet18', 'resnet50', 'vit_small', 'vit_base', 'deit_small', 'dbfn'],
                        help='Architecture du modèle (dbfn = DBFN complet ResNet-50 + ViT-Base)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint')

    parser.add_argument('--output', type=str, default=None,
                        help='Chemin pour sauvegarder le résultat (mode image)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device à utiliser')

    parser.add_argument('--benchmark_runs', type=int, default=1,
                        help="Nombre d'inférences pour moyenne (ex: 30, 50). 1 = mesure simple.")

    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print_system_info(device)

    model = load_model(args.model, args.checkpoint, device)
    face_cascade = load_face_detector()
    transform = get_transform()

    if args.image:
        test_image(args.image, model, transform, device, face_cascade, args.output, benchmark_runs=args.benchmark_runs)
    elif args.webcam:
        test_webcam(model, transform, device, face_cascade, benchmark_runs=args.benchmark_runs)


if __name__ == "__main__":
    main()
