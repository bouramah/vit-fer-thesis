ResNet-18 — meilleur run
Best epoch (Val F1) : epoch 66 (Val F1: 68.59%)
CUDA_VISIBLE_DEVICES=1 python train_v2.py \
  --model resnet18 \
  --device cuda \
  --batch_size 128 \
  --epochs 70 \
  --mixup 0.2 \
  --label_smoothing 0.1 \
  --dropout 0.4 \
  --checkpoint_dir checkpoints_run2 \
  --results_dir results_run2 \
  --log_dir logs_run2

Résultat (Test set)
Accuracy: 70.23%
Macro-F1: 68.99%


ResNet-50 — meilleur run
Best epoch (Val F1) : epoch 61 (Val F1: 69.76%)
CUDA_VISIBLE_DEVICES=2 python train_v2.py \
  --model resnet50 \
  --device cuda \
  --batch_size 64 \
  --epochs 80 \
  --two_stage \
  --lr 1e-4 \
  --mixup 0.1 \
  --label_smoothing 0.1 \
  --dropout 0.3 \
  --checkpoint_dir checkpoints_run3_resnet50 \
  --results_dir results_run3_resnet50 \
  --log_dir logs_run3_resnet50
  
Résultat (Test set)
Accuracy: 70.35%
Macro-F1: 69.78%

ViT-Small (retained — meilleur “small”)
Best epoch (Val F1) : epoch 67 (Val F1: 70.08%)
CUDA_VISIBLE_DEVICES=2 python train_vit_v3.py \
  --model vit_small \
  --epochs 80 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --drop_path 0.10 \
  --dropout 0.20 \
  --mixup 0.0 \
  --label_smoothing 0.05 \
  --weighted_sampler \
  --checkpoint_dir checkpoints_vit_small_v3_sampler_nomixup \
  --results_dir results_vit_small_v3_sampler_nomixup \
  --log_dir logs_vit_small_v3_sampler_nomixup

Résultat produit (Test set) :
Accuracy: 70.56%
Macro-F1: 69.67%



ViT-Base (retained — meilleur global)
🥇 Run retenu : vit_base v3 run1 (seed 42)
Best epoch (Val F1) : epoch 86 (Val F1: 71.10%)
CUDA_VISIBLE_DEVICES=1 python train_vit_v3.py \
  --model vit_base \
  --epochs 90 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --drop_path 0.18 \
  --dropout 0.20 \
  --mixup 0.10 \
  --label_smoothing 0.05 \
  --checkpoint_dir checkpoints_vit_base_v3_run1 \
  --results_dir results_vit_base_v3_run1 \
  --log_dir logs_vit_base_v3_run1 \
  --seed 42

Résultat produit (Test set) :
Accuracy: 72.64%
Macro-F1: 71.24%


DeiT (retained — deit_small)
CUDA_VISIBLE_DEVICES=2 python train_vit_v3.py \
  --model deit_small \
  --epochs 70 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --drop_path 0.15 \
  --dropout 0.20 \
  --mixup 0.10 \
  --label_smoothing 0.05 \
  --weighted_sampler \
  --checkpoint_dir checkpoints_deit_small_v3 \
  --results_dir results_deit_small_v3 \
  --log_dir logs_deit_small_v3

Résultat produit (Test set) :
Accuracy: 68.97%
Macro-F1: 67.82%