#!/usr/bin/env python3
"""Nettoyage checkpoints : liste blanche + prune au meilleur (val_f1) par dossier gardé."""
import os, glob, shutil, torch

# Dossiers à GARDER (chacun sera prune à son meilleur checkpoint)
KEEP = [
    # backbones + DBFN titre (FER2013)
    "checkpoints_run3_resnet50", "checkpoints_vit_base_v3_run1", "checkpoints_hybrid",
    # modèles individuels FER2013 retenus (best_models.sh)
    "checkpoints_run2",                          # ResNet-18
    "checkpoints_vit_small_v3_sampler_nomixup",  # ViT-Small
    "checkpoints_deit_small_v3",                 # DeiT-Small
    # multi-seed / ablation FER2013 (conservés à la demande)
    "checkpoints_ablation_concat", "checkpoints_ablation_gating", "checkpoints_ablation_cross_attention",
    "checkpoints_seed123_concat", "checkpoints_seed123_gating", "checkpoints_seed123_cross_attention",
    "checkpoints_seed2024_concat", "checkpoints_seed2024_gating", "checkpoints_seed2024_cross_attention",
    # RAF-DB
    "checkpoints_rafdb_resnet50", "checkpoints_rafdb_vit_base",
    "checkpoints_rafdb_fusion_seed42", "checkpoints_rafdb_fusion_seed123", "checkpoints_rafdb_fusion_seed2024",
    # AffectNet-7
    "checkpoints_affectnet_resnet50", "checkpoints_affectnet_vit_base",
    "checkpoints_affectnet_fusion_seed42", "checkpoints_affectnet_fusion_seed123", "checkpoints_affectnet_fusion_seed2024",
]
KEEP = set(KEEP)

def valf1(p):
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
        v = c.get("val_f1", None) if isinstance(c, dict) else None
        return v if v is not None else -1
    except Exception:
        return -2

all_dirs = sorted(d for d in glob.glob("checkpoints*") if os.path.isdir(d))
print("=== SUPPRESSION dossiers superseded ===")
for d in all_dirs:
    if d not in KEEP:
        sz = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(d) for f in fs)/1e9
        shutil.rmtree(d); print(f"  rm -rf {d}  (-{sz:.1f} Go)")

print("\n=== PRUNE au meilleur checkpoint (dossiers gardés) ===")
for d in sorted(KEEP):
    if not os.path.isdir(d):
        print(f"  [absent] {d}"); continue
    pth = glob.glob(os.path.join(d, "*.pth"))
    if len(pth) <= 1:
        print(f"  {d}: {len(pth)} ckpt (rien à prune)"); continue
    scored = [(valf1(p), '_best' in p, os.path.getmtime(p), p) for p in pth]
    scored.sort(key=lambda x:(x[0], x[1], x[2]))  # meilleur = dernier
    best = scored[-1][3]
    removed = 0
    for *_ , p in scored[:-1]:
        os.remove(p); removed += 1
    print(f"  {d}: gardé {os.path.basename(best)} (val_f1={scored[-1][0]:.2f}), supprimé {removed}")
print("\nOK.")
