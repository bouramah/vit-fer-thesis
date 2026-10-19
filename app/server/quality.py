"""Contrôle de la qualité d'un visage avant de faire confiance à la prédiction.

À partir du visage recadré et des cinq points YuNet (yeux, nez, coins de la bouche),
on mesure la netteté, la lumière, le bruit, la taille, l'orientation de la tête et
deux indices d'occultation (lunettes de soleil, masque). Chaque problème détecté
devient un avertissement affiché à côté de la prédiction ; il ne la bloque pas.

Les seuils sont calibrés avec app/tools/robustness_test.py sur FER2013 (voir README).
"""
import cv2
import numpy as np

# Seuils (calibration : app/tools/robustness_test.py)
MIN_FACE_PX = 40          # côté du visage détecté, en pixels de l'image d'origine
MIN_SHARPNESS = 18.0      # variance du laplacien, visage ramené à 112 px
MIN_BRIGHTNESS = 55.0     # moyenne des niveaux de gris (0-255)
MAX_BRIGHTNESS = 215.0
MIN_CONTRAST = 22.0       # écart-type des niveaux de gris
MAX_NOISE = 1.5           # écart-type du bruit dans les zones lisses (visage à 112 px) ; calibré sur photos réelles :
                          # 0,5 à 0,9 sans bruit ajouté (peau, barbe), 1,2 à 1,3 avec un bruit σ = 10, 2,2 à 2,7 avec σ = 25
MAX_YAW = 0.40            # décalage du nez / écart entre les yeux
MAX_ROLL = 30.0           # inclinaison de la ligne des yeux, en degrés
# Lunettes de soleil : yeux nettement plus sombres que le front, et sans détail
MAX_EYE_FOREHEAD = 0.35   # luminosité des yeux / du front
MAX_EYE_STD = 22.0        # écart-type des niveaux de gris dans les yeux
# Masque : bouche et joues de même luminosité, et bouche sans détail
MOUTH_CHEEK_BAND = (0.90, 1.05)  # luminosité bouche / joues
MAX_MOUTH_DETAIL = 0.55   # écart-type autour de la bouche / autour des yeux

SIZE = 112


def _noise_sigma(g):
    """Écart-type du bruit (méthode d'Immerkær) mesuré dans les zones les plus lisses du visage."""
    r = np.abs(cv2.filter2D(g, cv2.CV_32F, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)))[1:-1, 1:-1]
    grad = (np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)))[1:-1, 1:-1]
    flat = grad <= np.percentile(grad, 40)
    return float(np.sqrt(np.pi / 2) * r[flat].mean() / 6.0) if flat.any() else 0.0


def _box(gray, cx, cy, rx, ry):
    h, w = gray.shape
    x0, x1 = int(max(0, cx - rx)), int(min(w, cx + rx))
    y0, y1 = int(max(0, cy - ry)), int(min(h, cy + ry))
    return gray[y0:y1, x0:x1] if x1 - x0 > 2 and y1 - y0 > 2 else gray


def face_quality(face, lm=None, face_px=None):
    """face : image PIL du visage recadré ; lm : 5 points (x, y) dans le repère de face ;
    face_px : taille du visage dans l'image d'origine. Renvoie mesures et avertissements."""
    rgb = np.array(face.convert("RGB"))
    k = SIZE / rgb.shape[0]
    interp = cv2.INTER_AREA if rgb.shape[0] > SIZE else cv2.INTER_CUBIC
    gray = cv2.cvtColor(cv2.resize(rgb, (SIZE, SIZE), interpolation=interp), cv2.COLOR_RGB2GRAY)
    g = gray.astype(np.float32)
    q = {
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(g.mean()), 1),
        "contrast": round(float(g.std()), 1),
        "noise": round(_noise_sigma(g), 2),
        "face_px": round(float(face_px), 1) if face_px is not None else None,
    }
    if lm is not None:
        p = np.asarray(lm, np.float32).reshape(5, 2) * k
        reye, leye, nose, rmouth, lmouth = p
        d = max(1.0, float(np.linalg.norm(leye - reye)))
        mid = (reye + leye) / 2
        q["yaw"] = round(float((nose[0] - mid[0]) / d), 3)
        q["roll"] = round(float(np.degrees(np.arctan2(leye[1] - reye[1], leye[0] - reye[0]))), 1)
        eyes = np.concatenate([_box(g, *reye, 0.3 * d, 0.18 * d).ravel(), _box(g, *leye, 0.3 * d, 0.18 * d).ravel()])
        forehead = _box(g, mid[0], mid[1] - 0.6 * d, 0.6 * d, 0.2 * d)
        mouth = _box(g, (rmouth[0] + lmouth[0]) / 2, (rmouth[1] + lmouth[1]) / 2, 0.45 * d, 0.18 * d)
        cheeks = np.concatenate([_box(g, reye[0], nose[1] + 0.1 * d, 0.2 * d, 0.15 * d).ravel(),
                                 _box(g, leye[0], nose[1] + 0.1 * d, 0.2 * d, 0.15 * d).ravel()])
        eye_band = _box(g, mid[0], mid[1], 0.8 * d, 0.25 * d)
        q["eye_forehead"] = round(float(eyes.mean() / max(1.0, forehead.mean())), 3)
        q["eye_std"] = round(float(eyes.std()), 1)
        q["mouth_cheek"] = round(float(mouth.mean() / max(1.0, cheeks.mean())), 3)
        q["mouth_detail"] = round(float(mouth.std() / max(1.0, eye_band.std())), 3)

    w = []
    if face_px is not None and face_px < MIN_FACE_PX:
        w.append("small")
    if q["sharpness"] < MIN_SHARPNESS:
        w.append("blur")
    if q["brightness"] < MIN_BRIGHTNESS:
        w.append("dark")
    elif q["brightness"] > MAX_BRIGHTNESS:
        w.append("bright")
    elif q["contrast"] < MIN_CONTRAST:
        w.append("low_contrast")
    if q["noise"] > MAX_NOISE:
        w.append("noise")
    if lm is not None:
        if abs(q["yaw"]) > MAX_YAW or abs(q["roll"]) > MAX_ROLL:
            w.append("pose")
        if q["eye_forehead"] < MAX_EYE_FOREHEAD and q["eye_std"] < MAX_EYE_STD:
            w.append("eyes_hidden")
        if MOUTH_CHEEK_BAND[0] < q["mouth_cheek"] < MOUTH_CHEEK_BAND[1] and q["mouth_detail"] < MAX_MOUTH_DETAIL:
            w.append("mouth_hidden")
    q["warnings"] = w
    return q
