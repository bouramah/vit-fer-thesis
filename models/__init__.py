"""
Module Models - Architectures de réseaux de neurones pour FER
"""

from .resnet import ResNetFER, create_resnet18, create_resnet50
from .vit import (
    VisionTransformer,
    ViTForFER,
    create_vit_tiny,
    create_vit_small,
    create_vit_base,
    create_vit_pretrained
)

__all__ = [
    # ResNet
    'ResNetFER', 
    'create_resnet18', 
    'create_resnet50',
    # ViT
    'VisionTransformer',
    'ViTForFER',
    'create_vit_tiny',
    'create_vit_small',
    'create_vit_base',
    'create_vit_pretrained'
]
