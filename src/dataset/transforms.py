"""
Data Transforms for Surgical Video Frames.
"""

import torchvision.transforms as T
from typing import Optional, Dict, Any


def get_train_transforms(img_size: int = 224, augmentation_config: Optional[Dict] = None) -> T.Compose:
    """Get training transforms with augmentation."""
    if augmentation_config is None:
        augmentation_config = {
            'horizontal_flip': True,
            'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1},
            'random_rotation': 10,
            'random_crop': True,
        }
    
    transforms_list = [T.Resize((img_size + 32, img_size + 32))]
    
    if augmentation_config.get('random_crop', False):
        transforms_list.append(T.RandomCrop(img_size))
    else:
        transforms_list.append(T.CenterCrop(img_size))
    
    if augmentation_config.get('horizontal_flip', False):
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
    
    cj = augmentation_config.get('color_jitter')
    if cj:
        transforms_list.append(T.ColorJitter(**cj))
    
    rot = augmentation_config.get('random_rotation', 0)
    if rot > 0:
        transforms_list.append(T.RandomRotation(rot))
    
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(transforms_list)


def get_val_transforms(img_size: int = 224) -> T.Compose:
    """Get validation transforms (no augmentation)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
