"""Moteur d'inférence d'EmoVision : ResNet-50, ViT-Base et DBFN.

Les trois modèles partagent le même prétraitement que l'entraînement
(niveaux de gris répliqués sur 3 canaux, 224×224, normalisation ImageNet).
En mode « compare », le ViT est exécuté une seule fois : ses caractéristiques
servent à la fois au ViT seul et au DBFN.
"""
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models as tvm
from torchvision import transforms

from models.hybrid import ConcatFusion

MODELS = ("resnet", "vit", "dbfn")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _resnet50():
    m = tvm.resnet50(weights=None)
    m.fc = nn.Identity()
    return m


def _load(module, sd, prefix=""):
    sub = {k[len(prefix):]: v.float() if v.is_floating_point() else v for k, v in sd.items() if k.startswith(prefix)}
    module.load_state_dict(sub, strict=True)
    return module


class OnnxBackbones:
    """Réseaux lourds exécutés par ONNX Runtime (voir app/tools/export_onnx.py)."""

    def __init__(self, onnx_dir, threads, int8_vit=False):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        mk = lambda name: ort.InferenceSession(os.path.join(onnx_dir, name), opts, providers=["CPUExecutionProvider"])
        self.resnet = mk("resnet.onnx")
        self.vit = mk("vit_feat_int8.onnx" if int8_vit else "vit_feat.onnx")
        self.dbfn_resnet = mk("dbfn_resnet.onnx")

    @staticmethod
    def run(sess, x):
        return torch.from_numpy(sess.run(None, {"x": x.numpy()})[0])


class Engine:
    def __init__(self, weights_path, threads=None, device="cpu", backend="torch", onnx_dir=None):
        self.device = torch.device(device)
        self.backend = backend
        torch.set_num_threads(threads or max(1, os.cpu_count() or 1))
        try:
            torch.set_num_interop_threads(1)  # évite des threads concurrents (quota CPU des conteneurs)
        except RuntimeError:
            pass  # déjà fixé dans ce processus
        pack = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.classes = pack["classes"]

        self.resnet = _load(_resnet50(), pack["resnet"], "backbone.")
        self.resnet_head = _load(nn.Linear(2048, 7), pack["resnet"], "classifier.1.")

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, drop_path_rate=0.0)
        _load(self.vit, pack["vit"], "backbone.")
        self.vit_head = _load(nn.Linear(768, 7), pack["vit"], "classifier.1.")

        self.dbfn_resnet = _load(_resnet50(), pack["dbfn_resnet"])
        self.fusion = _load(ConcatFusion(dropout=0.2), pack["dbfn_fusion"])

        for m in (self.resnet, self.resnet_head, self.vit, self.vit_head, self.dbfn_resnet, self.fusion):
            m.eval().to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)

        self.onnx = None
        if backend in ("onnx", "onnx-int8"):
            onnx_dir = onnx_dir or os.path.join(os.path.dirname(weights_path), "onnx")
            self.onnx = OnnxBackbones(onnx_dir, threads or max(1, os.cpu_count() or 1), int8_vit=backend == "onnx-int8")

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def batch(self, faces):
        return torch.stack([self.tf(f.convert("RGB")) for f in faces]).to(self.device)

    @torch.inference_mode()
    def predict(self, faces, model="dbfn"):
        """faces : liste d'images PIL de visages recadrés.
        Renvoie {nom_modèle: tableau [N, 7] de probabilités} et la latence en ms."""
        if not faces:
            return {}, 0.0
        x = self.batch(faces)
        t0 = time.perf_counter()
        out = {}
        need_vit = model in ("vit", "dbfn", "compare")
        if self.onnx:
            o = self.onnx
            v = o.run(o.vit, x) if need_vit else None
            if model in ("resnet", "compare"):
                out["resnet"] = o.run(o.resnet, x)
            if model in ("vit", "compare"):
                out["vit"] = self.vit_head(v)
            if model in ("dbfn", "compare"):
                out["dbfn"] = self.fusion(o.run(o.dbfn_resnet, x), v)
        else:
            v = self.vit(x) if need_vit else None
            if model in ("resnet", "compare"):
                out["resnet"] = self.resnet_head(self.resnet(x))
            if model in ("vit", "compare"):
                out["vit"] = self.vit_head(v)
            if model in ("dbfn", "compare"):
                out["dbfn"] = self.fusion(self.dbfn_resnet(x), v)
        ms = (time.perf_counter() - t0) * 1000
        return {k: F.softmax(t, dim=1).cpu().numpy() for k, t in out.items()}, ms

    # ------------------------------------------------------------------ explicabilité
    @staticmethod
    def _norm(cam):
        cam = cam - cam.min()
        return cam / (cam.max() + 1e-8)

    def _cam_hooks(self, layer):
        """Capture les activations et gradients d'un étage convolutif (Grad-CAM)."""
        store = {}
        h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
        return store, (h1, h2)

    def _cam_from(self, store):
        w = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * store["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        return self._norm(cam).detach().cpu().numpy()

    def _attn_hooks(self):
        """Capture les cartes d'attention de chaque bloc du ViT et leurs gradients."""
        maps, hooks = [], []
        for blk in self.vit.blocks:
            blk.attn.fused_attn = False  # matrice d'attention explicite (le calcul fusionné ne l'expose pas)

            def fwd(m, i, o):
                o.retain_grad()
                maps.append(o)
            hooks.append(blk.attn.attn_drop.register_forward_hook(fwd))
        return maps, hooks

    def _rollout_from(self, maps):
        """Attribution du ViT propre à la classe (Chefer et al., 2021) :
        attention pondérée par son gradient, propagée bloc par bloc."""
        n = maps[0].shape[-1]
        r = torch.eye(n)
        for a in maps:
            g = a.grad[0]
            cam = (g * a[0]).clamp(min=0).mean(0).detach()
            r = r + cam @ r
        side = int(round((n - 1) ** 0.5))
        m = r[0, 1:].reshape(side, side).numpy()
        # Quelques patchs isolés, souvent dans le fond, concentrent l'attention sans rapport avec le visage
        # (artefact connu des ViT : Darcet et al., 2024). Un filtre médian 3×3 sur la grille des patchs
        # retire ces points isolés et garde les zones cohérentes.
        import cv2
        m = cv2.medianBlur(m.astype(np.float32), 3)
        m = torch.from_numpy(m).reshape(1, 1, side, side)
        m = F.interpolate(m, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        return self._norm(m).cpu().numpy()

    def explain(self, face, model="dbfn", cls=None):
        """Explication de la décision pour un visage.
        resnet : Grad-CAM sur le dernier étage ; vit : attribution d'attention pondérée par le gradient ;
        dbfn : les deux cartes (une par branche), leur part dans la décision et une carte combinée."""
        model = "dbfn" if model == "compare" else model
        x = self.batch([face])
        out = {}
        hooks = []
        try:
            with torch.enable_grad():
                x.requires_grad_(True)  # les poids sont gelés : le gradient part de l'image
                if model == "resnet":
                    store, hooks = self._cam_hooks(self.resnet.layer4)
                    logits = self.resnet_head(self.resnet(x))
                    cls = int(logits.argmax(1)) if cls is None else cls
                    logits[0, cls].backward()
                    out["cnn"] = self._cam_from(store)
                elif model == "vit":
                    maps, hooks = self._attn_hooks()
                    logits = self.vit_head(self.vit(x))
                    cls = int(logits.argmax(1)) if cls is None else cls
                    logits[0, cls].backward()
                    out["vit"] = self._rollout_from(maps)
                else:
                    store, h1 = self._cam_hooks(self.dbfn_resnet.layer4)
                    maps, h2 = self._attn_hooks()
                    hooks = list(h1) + h2
                    fc = self.dbfn_resnet(x)
                    fv = self.vit(x)
                    fc.retain_grad()
                    fv.retain_grad()
                    logits = self.fusion(fc, fv)
                    cls = int(logits.argmax(1)) if cls is None else cls
                    logits[0, cls].backward()
                    out["cnn"] = self._cam_from(store)
                    out["vit"] = self._rollout_from(maps)
                    # Part de chaque branche : |gradient × caractéristique| sommé sur son vecteur
                    sc = float((fc.grad * fc).abs().sum().detach())
                    sv = float((fv.grad * fv).abs().sum().detach())
                    share = sc / max(1e-8, sc + sv)
                    out["share"] = {"cnn": round(share, 3), "vit": round(1 - share, 3)}
                    out["combined"] = self._norm(share * out["cnn"] + (1 - share) * out["vit"])
        finally:
            for h in hooks:
                h.remove()
            for blk in self.vit.blocks:
                blk.attn.fused_attn = True
        return out, cls

    def gradcam(self, face, cls=None):
        """Grad-CAM de la branche ResNet-50 du DBFN (compatibilité)."""
        out, cls = self.explain(face, "dbfn", cls)
        return out["cnn"], cls


def overlay(face, cam, alpha=0.45):
    """Superpose une carte de chaleur (palette jet simplifiée) sur le visage."""
    import cv2
    base = np.array(face.convert("RGB"))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return Image.fromarray((base * (1 - alpha) + heat * alpha).astype(np.uint8))
