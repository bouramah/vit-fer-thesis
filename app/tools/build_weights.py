"""Assemble les poids d'EmoVision en un seul fichier compact.

Le fichier contient ResNet-50 et ViT-Base (modèles seuls, avec leur tête) et,
pour le DBFN, uniquement ce qui diffère : la branche ResNet (ses statistiques
de BatchNorm ont été ré-estimées pendant l'entraînement de la fusion) et la
tête de fusion. Le ViT du DBFN est strictement identique au ViT seul : il n'est
stocké qu'une fois. Les tenseurs flottants sont enregistrés en float16.

Usage (depuis la racine du dépôt) :
    python app/tools/build_weights.py \
        --resnet checkpoints_run3_resnet50/resnet50_v2_epoch61_best.pth \
        --vit checkpoints_vit_base_v3_run1/vit_base_v3_epoch86_best.pth \
        --dbfn checkpoints_hybrid/dbfn_concat_frozen_20260227_173625_epoch1_best.pth \
        --out app/weights/emovision_weights.pt
"""
import argparse

import torch

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt


def half(sd):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", required=True)
    ap.add_argument("--vit", required=True)
    ap.add_argument("--dbfn", required=True)
    ap.add_argument("--out", default="app/weights/emovision_weights.pt")
    args = ap.parse_args()

    resnet, vit, dbfn = state_dict(args.resnet), state_dict(args.vit), state_dict(args.dbfn)

    shared = [k for k in vit if k.startswith("backbone.")]
    for k in shared:
        assert torch.equal(vit[k], dbfn["vit_branch.vit." + k[len("backbone."):]]), k

    pack = {
        "version": 1,
        "classes": CLASSES,
        "resnet": half(resnet),
        "vit": half(vit),
        "dbfn_resnet": half({k[len("cnn_branch.resnet."):]: v for k, v in dbfn.items() if k.startswith("cnn_branch.resnet.")}),
        "dbfn_fusion": half({k[len("fusion."):]: v for k, v in dbfn.items() if k.startswith("fusion.")}),
    }
    torch.save(pack, args.out)
    print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
