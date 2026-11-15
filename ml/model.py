"""U-Net with a ResNet34 encoder, and the image preprocessing shared by training and inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from .data import WEIGHTS, to_rgb

INPUT_SIZE = 256
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(pretrained: bool = True) -> torch.nn.Module:
    """Four-class U-Net; the encoder starts from ImageNet weights."""
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, classes=4)


def preprocess(image, size: int = INPUT_SIZE) -> torch.Tensor:
    """uint8 grayscale/RGB image -> ImageNet-normalised (3, size, size) tensor."""
    img = to_rgb(image)
    interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_LINEAR
    img = cv2.resize(img, (size, size), interpolation=interp).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def load_weights(path: str | Path = WEIGHTS, device="cpu") -> torch.nn.Module:
    """Load the exported fp16 weights into an fp32 model in eval mode."""
    if not Path(path).exists():
        raise FileNotFoundError(f"no weights at {path}; run `python -m ml.train` or set RR_WEIGHTS")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.float().to(device).eval()
