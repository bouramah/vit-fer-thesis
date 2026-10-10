"""Exporte les réseaux lourds d'EmoVision au format ONNX (accélération CPU avec ONNX Runtime).

Trois graphes sont produits dans app/weights/onnx/ :
    resnet.onnx       ResNet-50 seul, tête comprise          -> logits [N, 7]
    vit_feat.onnx     ViT-Base, caractéristiques CLS          -> [N, 768]
    dbfn_resnet.onnx  branche ResNet-50 du DBFN               -> [N, 2048]
Les petites têtes (ViT et fusion du DBFN) restent en PyTorch.
Option --quantize : version int8 dynamique du ViT (vit_feat_int8.onnx).

Usage (depuis la racine du dépôt) :
    python app/tools/export_onnx.py [--quantize]
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.server.engine import Engine  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "weights" / "onnx"


class ResNetFull(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.backbone, self.head = e.resnet, e.resnet_head

    def forward(self, x):
        return self.head(self.backbone(x))


def export(module, name):
    x = torch.randn(1, 3, 224, 224)
    path = OUT / name
    torch.onnx.export(module.eval(), (x,), str(path), input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, opset_version=17, dynamo=False)
    print(f"Écrit : {path} ({path.stat().st_size / 1e6:.0f} Mo)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(OUT.parent / "emovision_weights.pt"))
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    e = Engine(args.weights, threads=4)
    with torch.no_grad():
        export(ResNetFull(e), "resnet.onnx")
        vit = export(e.vit, "vit_feat.onnx")
        export(e.dbfn_resnet, "dbfn_resnet.onnx")
    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = OUT / "vit_feat_int8.onnx"
        quantize_dynamic(str(vit), str(q), weight_type=QuantType.QInt8)
        print(f"Écrit : {q} ({q.stat().st_size / 1e6:.0f} Mo)")


if __name__ == "__main__":
    main()
