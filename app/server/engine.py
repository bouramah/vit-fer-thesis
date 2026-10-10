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

    def gradcam(self, face, cls=None):
        """Grad-CAM sur le dernier étage (layer4) du ResNet-50 du DBFN.
        Renvoie une carte 224×224 normalisée dans [0, 1] et la classe expliquée."""
        x = self.batch([face])
        acts, grads = {}, {}
        layer = self.dbfn_resnet.layer4
        h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
        try:
            with torch.enable_grad():
                x.requires_grad_(True)
                feats = self.dbfn_resnet(x)
                with torch.no_grad():
                    v = self.vit(x)
                logits = self.fusion(feats, v)
                cls = int(logits.argmax(1)) if cls is None else cls
                logits[0, cls].backward()
        finally:
            h1.remove()
            h2.remove()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), cls


def overlay(face, cam, alpha=0.45):
    """Superpose une carte de chaleur (palette jet simplifiée) sur le visage."""
    import cv2
    base = np.array(face.convert("RGB"))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return Image.fromarray((base * (1 - alpha) + heat * alpha).astype(np.uint8))
