"""Stockage des contributions consenties et des retours anonymes.

Chaque contribution est un dossier <DATA_DIR>/contributions/<id>/ contenant
meta.json (consentement, prédictions, étiquettes) et les images JPEG.
Aucune image n'est conservée sans consentement explicite.
"""
import base64
import io
import json
import secrets
import shutil
import threading
import time
from pathlib import Path

from PIL import Image

CONSENT_VERSION = "2026-10-v1"
MAX_ITEMS = 12
CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
MODELS = ("resnet", "vit", "dbfn")
SOURCES = ("photo", "snapshot", "live", "challenge")


def _label(v):
    return v if v in CLASSES else None


def _source(v):
    return v if v in SOURCES else None


def _predictions(p):
    """Ne conserve que des prédictions bien formées (7 probabilités par modèle)."""
    out = {}
    for m in MODELS:
        r = (p or {}).get(m) if isinstance(p, dict) else None
        try:
            probs = [round(float(x), 4) for x in r["probs"]][:7]
        except (TypeError, KeyError, ValueError):
            continue
        if len(probs) == 7:
            best = max(range(7), key=probs.__getitem__)
            out[m] = {"probs": probs, "top": CLASSES[best], "confidence": probs[best]}
    return out


class Storage:
    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.contrib = self.root / "contributions"
        self.contrib.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.root / "feedback.jsonl"
        self._lock = threading.Lock()

    # ---------------------------------------------------------- contributions
    def save_contribution(self, payload):
        consent = payload.get("consent") or {}
        if not (consent.get("memoire") and consent.get("adult") and consent.get("self")):
            raise ValueError("consent_required")
        items = payload.get("items") or []
        if not 1 <= len(items) <= MAX_ITEMS:
            raise ValueError("bad_items")

        cid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        code = "-".join(secrets.token_hex(2).upper() for _ in range(3))
        folder = self.contrib / cid
        folder.mkdir(parents=True)
        stored = []
        for i, it in enumerate(items):
            img = decode_image(it["image"], max_side=1024)
            name = f"img_{i:02d}.jpg"
            img.save(folder / name, quality=92)
            stored.append({
                "file": name,
                "source": _source(it.get("source")),
                "target": _label(it.get("target")),
                "true_label": _label(it.get("true_label")),
                "predictions": _predictions(it.get("predictions")),
                "success": bool(it.get("success")) if it.get("success") is not None else None,
            })
        meta = {
            "id": cid,
            "withdrawal_code_hash": _hash(code),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "consent": {
                "version": CONSENT_VERSION,
                "memoire": bool(consent.get("memoire")),
                "publications": bool(consent.get("publications")),
                "adult": bool(consent.get("adult")),
                "self_or_all_agreed": bool(consent.get("self")),
                "lang": consent.get("lang", "fr"),
            },
            "pseudonym": (payload.get("pseudonym") or "")[:60],
            "contact": (payload.get("contact") or "")[:120],
            "items": stored,
        }
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return cid, code

    def withdraw(self, code):
        h = _hash(code.strip().upper())
        for meta_path in self.contrib.glob("*/meta.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if secrets.compare_digest(meta.get("withdrawal_code_hash", ""), h):
                shutil.rmtree(meta_path.parent)
                return True
        return False

    def list(self):
        out = []
        for meta_path in sorted(self.contrib.glob("*/meta.json"), reverse=True):
            out.append(json.loads(meta_path.read_text(encoding="utf-8")))
        return out

    def delete(self, cid):
        folder = self.contrib / cid
        if folder.parent == self.contrib and folder.is_dir():
            shutil.rmtree(folder)
            return True
        return False

    def export_zip(self):
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in self.contrib.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(self.root))
            if self.feedback_file.exists():
                z.write(self.feedback_file, self.feedback_file.name)
        return buf.getvalue()

    # ---------------------------------------------------------- retours anonymes
    def add_feedback(self, fb):
        predicted, true_label = _label(fb.get("predicted")), _label(fb.get("true_label"))
        if not (predicted and true_label):
            return
        rec = {
            "t": int(time.time()),
            "model": fb.get("model") if fb.get("model") in MODELS else None,
            "predicted": predicted,
            "true_label": true_label,
            "correct": predicted == true_label,
            "source": _source(fb.get("source")),
        }
        with self._lock, self.feedback_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def stats(self):
        n_fb = n_ok = 0
        if self.feedback_file.exists():
            for line in self.feedback_file.read_text(encoding="utf-8").splitlines():
                n_fb += 1
                n_ok += json.loads(line).get("correct", False)
        return {
            "contributions": sum(1 for _ in self.contrib.glob("*/meta.json")),
            "feedback": n_fb,
            "feedback_correct": n_ok,
        }


def _hash(code):
    import hashlib
    return hashlib.sha256(code.encode()).hexdigest()


def decode_image(data, max_side=None):
    """Décode une image envoyée en base64 (avec ou sans en-tête data:)."""
    if "," in data[:80]:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("image_too_large")
    img = Image.open(io.BytesIO(raw))
    img.load()
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img
