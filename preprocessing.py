from pathlib import Path
from typing import List, Sequence, Union

import cv2
import numpy as np
import torch

from config import DEVICE, IMG_SIZE, NUM_CLASSES, PREDICTION_THRESHOLD


def decode_channel_bytes(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid PNG/JPG.")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def load_channel_image(path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read channel image at '{path}'.")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def stack_channels(channels: Sequence[np.ndarray]) -> np.ndarray:
    if len(channels) != 4:
        raise ValueError("Exactly four channels are required: red, green, blue, yellow.")
    expected = (IMG_SIZE, IMG_SIZE)
    for channel in channels:
        if np.asarray(channel).shape != expected:
            raise ValueError(
                f"Each channel must have shape {expected}; got {np.asarray(channel).shape}."
            )
    return np.stack(channels, axis=0).astype(np.float32) / 255.0


def channels_to_tensor(
    channels: Sequence[np.ndarray],
    device: torch.device = DEVICE,
) -> torch.Tensor:
    return torch.from_numpy(stack_channels(channels)).unsqueeze(0).to(device)


def predict_from_channels(
    model,
    channels: List[np.ndarray],
    threshold: float = PREDICTION_THRESHOLD,
    device: torch.device = DEVICE,
):
    tensor = channels_to_tensor(channels, device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs > threshold).astype(int)
    return probs.tolist(), preds.tolist()


def create_multilabel(target, num_classes: int = NUM_CLASSES) -> np.ndarray:
    label = np.zeros(num_classes, dtype=np.float32)
    for token in str(target).split():
        label[int(token)] = 1.0
    return label
