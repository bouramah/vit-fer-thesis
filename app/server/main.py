"""EmoVision — API et serveur web.

Variables d'environnement :
  EMOVISION_WEIGHTS   chemin du fichier de poids (app/weights/emovision_weights.pt)
  EMOVISION_YUNET     chemin du détecteur YuNet (.onnx)
  EMOVISION_DATA      dossier des contributions consenties (app/data)
  EMOVISION_THREADS   nombre de threads de calcul
  EMOVISION_BACKEND   torch (défaut), onnx ou onnx-int8 (ONNX Runtime, plus rapide sur CPU)
  ADMIN_PASSWORD      mot de passe de la page /admin (désactivée si vide)
  CONTACT_EMAIL       adresse affichée pour les demandes de retrait
"""
import base64
import html
import io
import os
import secrets
import threading
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .engine import Engine, overlay
from .faces import CROP_DY, CROP_SCALE, FaceDetector, crop_box, crop_face
from .quality import face_quality
from .storage import CONSENT_VERSION, Storage, decode_image

APP_DIR = Path(__file__).resolve().parent.parent
WEIGHTS = os.environ.get("EMOVISION_WEIGHTS", str(APP_DIR / "weights" / "emovision_weights.pt"))
YUNET = os.environ.get("EMOVISION_YUNET", str(APP_DIR / "weights" / "face_detection_yunet_2023mar.onnx"))
DATA = os.environ.get("EMOVISION_DATA", str(APP_DIR / "data"))
THREADS = int(os.environ.get("EMOVISION_THREADS", "0")) or None
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
MAX_FACES = 8
VALID_MODELS = ("resnet", "vit", "dbfn", "compare")

app = FastAPI(title="EmoVision", docs_url=None, redoc_url=None)
BACKEND = os.environ.get("EMOVISION_BACKEND", "torch")  # torch | onnx | onnx-int8
engine = Engine(WEIGHTS, threads=THREADS, backend=BACKEND)
detector = FaceDetector(YUNET)
storage = Storage(DATA)
infer_lock = threading.Lock()
security = HTTPBasic()


def _results(probs, n):
    out = []
    for i in range(n):
        res = {}
        for m, p in probs.items():
            res[m] = {"probs": [round(float(v), 4) for v in p[i]], "top": engine.classes[int(p[i].argmax())],
                      "confidence": round(float(p[i].max()), 4)}
        out.append(res)
    return out


def _predict(faces, model):
    if model not in VALID_MODELS:
        raise HTTPException(400, "bad_model")
    with infer_lock:
        probs, ms = engine.predict(faces, model)
    return _results(probs, len(faces)), ms


def _b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=88)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ API publique
@app.get("/api/config")
def config():
    return {"classes": engine.classes, "models": list(VALID_MODELS), "crop_scale": CROP_SCALE, "crop_dy": CROP_DY,
            "max_faces": MAX_FACES, "consent_version": CONSENT_VERSION, "contact_email": CONTACT_EMAIL}


@app.post("/api/predict")
async def predict(request: Request):
    """Vignettes de visages déjà recadrées par le navigateur."""
    body = await request.json()
    faces = [decode_image(f, max_side=448) for f in (body.get("faces") or [])[:MAX_FACES]]
    results, ms = _predict(faces, body.get("model", "dbfn"))
    return {"results": results, "latency_ms": round(ms, 1)}


@app.post("/api/analyze")
async def analyze(request: Request):
    """Image complète : détection YuNet côté serveur puis prédiction."""
    body = await request.json()
    img = decode_image(body["image"], max_side=1600)
    boxes = detector.detect(np.array(img))[:MAX_FACES]
    faces = [crop_face(img, *b[:4]) for b in boxes]
    results, ms = _predict(faces, body.get("model", "dbfn")) if faces else ([], 0.0)
    quality = []
    for b, face in zip(boxes, faces):
        left, top, _ = crop_box(*b[:4])
        pts = [(x - left, y - top) for x, y in b[5]]  # points dans le repère du visage recadré
        quality.append(face_quality(face, pts, face_px=max(b[2], b[3])))
    return {
        "width": img.width, "height": img.height, "latency_ms": round(ms, 1),
        "faces": [{"box": [round(v, 1) for v in b[:4]], "score": round(b[4], 3), "quality": q, **r}
                  for b, q, r in zip(boxes, quality, results)],
    }


@app.post("/api/explain")
async def explain(request: Request):
    body = await request.json()
    face = decode_image(body["face"], max_side=448)
    model = body.get("model", "dbfn")
    if model not in VALID_MODELS:
        raise HTTPException(400, "bad_model")
    with infer_lock:
        maps, cls = engine.explain(face, model)
    base = face.resize((224, 224))
    res = {"class": engine.classes[cls], "model": "dbfn" if model == "compare" else model,
           "maps": {k: _b64(overlay(base, v)) for k, v in maps.items() if k != "share"}}
    if "share" in maps:
        res["share"] = maps["share"]
    res["overlay"] = res["maps"].get("combined") or next(iter(res["maps"].values()))
    return res


@app.post("/api/contribute")
async def contribute(request: Request):
    body = await request.json()
    try:
        cid, code = storage.save_contribution(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": cid, "withdrawal_code": code}


@app.post("/api/withdraw")
async def withdraw(request: Request):
    body = await request.json()
    return {"deleted": storage.withdraw(body.get("code", ""))}


@app.post("/api/feedback")
async def feedback(request: Request):
    storage.add_feedback(await request.json())
    return {"ok": True}


@app.get("/api/stats")
def stats():
    return storage.stats()


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------ administration
def admin(credentials: HTTPBasicCredentials = Depends(security)):
    ok = ADMIN_PASSWORD and secrets.compare_digest(credentials.password.encode(), ADMIN_PASSWORD.encode())
    if not ok:
        raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})
    return True


@app.get("/admin", response_class=HTMLResponse)
def admin_page(_=Depends(admin)):
    rows = []
    for m in storage.list():
        e = html.escape
        thumbs = "".join(f'<img src="/admin/img/{e(m["id"])}/{e(it["file"])}" title="{e(str(it.get("target") or it.get("true_label") or ""))}">'
                         for it in m["items"])
        c = m["consent"]
        rows.append(f'<tr><td>{e(m["id"])}<br><small>{e(m["created"])}</small></td><td>{e(m["pseudonym"])}<br><small>{e(m["contact"])}</small></td>'
                    f'<td>mémoire : {"oui" if c["memoire"] else "non"}<br>publications : {"oui" if c["publications"] else "non"}<br>v{c["version"]}</td>'
                    f'<td class="t">{thumbs}</td><td><form method="post" action="/admin/delete/{m["id"]}" onsubmit="return confirm(\'Supprimer ?\')">'
                    f'<button>Supprimer</button></form></td></tr>')
    s = storage.stats()
    return f"""<!doctype html><meta charset="utf-8"><title>EmoVision — admin</title>
<style>body{{font-family:system-ui;margin:24px}}td{{border-bottom:1px solid #ddd;padding:8px;vertical-align:top}}
.t img{{height:72px;margin:2px;border-radius:6px}}</style>
<h1>EmoVision — contributions</h1><p>{s["contributions"]} contributions · {s["feedback"]} retours ({s["feedback_correct"]} justes)
· <a href="/admin/export.zip">Exporter tout (zip)</a></p><table>{''.join(rows)}</table>"""


@app.get("/admin/img/{cid}/{name}")
def admin_img(cid: str, name: str, _=Depends(admin)):
    p = (storage.contrib / cid / name).resolve()
    if storage.contrib.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/admin/delete/{cid}")
def admin_delete(cid: str, _=Depends(admin)):
    storage.delete(cid)
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/admin">')


@app.get("/admin/export.zip")
def admin_export(_=Depends(admin)):
    return Response(storage.export_zip(), media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=emovision_contributions.zip"})


# ------------------------------------------------------------------ frontend
@app.middleware("http")
async def no_cache_frontend(request: Request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory=APP_DIR / "web", html=True), name="web")
