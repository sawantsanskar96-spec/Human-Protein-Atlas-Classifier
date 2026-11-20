import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _upload_files():
    rng = np.random.default_rng(7)
    files = {}
    for name in ("red", "green", "blue", "yellow"):
        array = rng.integers(0, 256, (config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", array)
        assert ok
        files[name] = (f"{name}.png", buffer.tobytes(), "image/png")
    return files


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Human Protein Classifier API" in response.json()["message"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_classes(client):
    response = client.get("/classes")
    assert response.status_code == 200
    assert response.json() == {str(i): label for i, label in enumerate(config.CLASS_LABELS)}


def test_predict(client):
    response = client.post("/predict", files=_upload_files(), params={"threshold": 0.3})
    assert response.status_code == 200

    payload = response.json()
    assert set(payload["probabilities"]) == set(config.CLASS_LABELS)
    assert len(payload["probabilities"]) == config.NUM_CLASSES
    assert payload["threshold_used"] == 0.3
    assert 0 <= len(payload["class_indices"]) <= config.NUM_CLASSES
    assert all(0 <= idx < config.NUM_CLASSES for idx in payload["class_indices"])
    assert len(payload["predicted_classes"]) == len(payload["class_indices"])


def test_predict_rejects_invalid_threshold(client):
    response = client.post("/predict", files=_upload_files(), params={"threshold": 1.5})
    assert response.status_code == 422


def test_predict_rejects_invalid_image(client):
    files = _upload_files()
    files["red"] = ("red.png", b"not an image", "image/png")
    response = client.post("/predict", files=files)
    assert response.status_code == 400
