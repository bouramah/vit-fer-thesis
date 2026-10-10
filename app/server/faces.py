"""Détection de visages côté serveur (YuNet, OpenCV) et recadrage commun.

Le recadrage reproduit le cadrage des images d'entraînement (FER2013 : front,
cheveux et menton visibles) : carré de côté CROP_SCALE × max(largeur, hauteur)
du cadre détecté, centre légèrement remonté de CROP_DY × hauteur.
"""
import threading

import cv2
import numpy as np
from PIL import Image

CROP_SCALE = 1.3
CROP_DY = -0.05


class FaceDetector:
    def __init__(self, model_path, score_threshold=0.7):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self._lock = threading.Lock()
        self._det = None

    def detect(self, rgb):
        """rgb : tableau numpy HxWx3. Renvoie une liste de (x, y, w, h, score)."""
        h, w = rgb.shape[:2]
        scale = min(1.0, 1280 / max(h, w))
        img = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1 else rgb
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with self._lock:
            if self._det is None:
                self._det = cv2.FaceDetectorYN.create(self.model_path, "", (bgr.shape[1], bgr.shape[0]), self.score_threshold)
            self._det.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = self._det.detect(bgr)
        if faces is None:
            return []
        return [(float(f[0] / scale), float(f[1] / scale), float(f[2] / scale), float(f[3] / scale), float(f[14])) for f in faces]


def crop_box(x, y, w, h, scale=CROP_SCALE, dy=CROP_DY):
    side = max(w, h) * scale
    cx, cy = x + w / 2, y + h / 2 + dy * h
    return cx - side / 2, cy - side / 2, side


def crop_face(img: Image.Image, x, y, w, h):
    """Recadre un visage (avec remplissage noir si le carré dépasse l'image)."""
    left, top, side = crop_box(x, y, w, h)
    return img.crop((int(left), int(top), int(left + side), int(top + side)))
