from typing import Any, Dict, Iterable, List, Tuple

import requests

from config import CLASS_LABELS, NUM_CLASSES

ChannelFiles = Dict[str, Any]


def _to_upload_tuple(name: str, channel_file: Any) -> Tuple[str, bytes, str]:
    if hasattr(channel_file, "getvalue"):
        filename = getattr(channel_file, "name", f"{name}.png")
        content_type = getattr(channel_file, "type", None) or "image/png"
        return filename, channel_file.getvalue(), content_type
    if isinstance(channel_file, tuple):
        filename = channel_file[0] or f"{name}.png"
        content = channel_file[1]
        content_type = channel_file[2] if len(channel_file) > 2 else "image/png"
        return filename, content, content_type or "image/png"
    if isinstance(channel_file, bytes):
        return f"{name}.png", channel_file, "image/png"
    filename = getattr(channel_file, "name", f"{name}.png")
    content_type = getattr(channel_file, "type", None) or "image/png"
    return filename, channel_file.read(), content_type


def check_api_health(api_url: str, timeout: float = 5.0) -> bool:
    response = requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
    response.raise_for_status()
    return response.status_code == 200


def predict_via_api(
    api_url: str,
    channel_files: ChannelFiles,
    threshold: float,
    timeout: float = 30.0,
) -> Tuple[List[float], List[int]]:
    files = {
        name: _to_upload_tuple(name, channel_files[name])
        for name in ("red", "green", "blue", "yellow")
    }
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        files=files,
        params={"threshold": threshold},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    probabilities_by_label: Dict[str, float] = payload["probabilities"]
    probs = [float(probabilities_by_label[label]) for label in CLASS_LABELS]
    preds = [0] * NUM_CLASSES
    for index in payload["class_indices"]:
        preds[int(index)] = 1
    return probs, preds
