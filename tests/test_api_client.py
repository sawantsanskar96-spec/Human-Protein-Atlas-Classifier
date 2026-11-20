import api_client
from config import CLASS_LABELS, NUM_CLASSES


class FakeResponse:
    status_code = 200

    def __init__(self, payload=None):
        self.payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeUpload:
    name = "red.png"
    type = "image/png"

    def getvalue(self):
        return b"png-bytes"


def test_check_api_health_uses_service_hostname(monkeypatch):
    captured = {}

    def fake_get(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    assert api_client.check_api_health("http://fastapi:8000/") is True
    assert captured["url"] == "http://fastapi:8000/health"


def test_predict_via_api_maps_response_to_class_order(monkeypatch):
    captured = {}
    payload = {
        "probabilities": {
            label: 0.01 + index / 100 for index, label in enumerate(CLASS_LABELS)
        },
        "class_indices": [0, NUM_CLASSES - 1],
    }

    def fake_post(url, files, params, timeout):
        captured["url"] = url
        captured["files"] = files
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse(payload)

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    files = {name: FakeUpload() for name in ("red", "green", "blue", "yellow")}
    probs, preds = api_client.predict_via_api("http://fastapi:8000", files, 0.3)

    assert captured["url"] == "http://fastapi:8000/predict"
    assert captured["params"]["threshold"] == 0.3
    assert set(captured["files"]) == {"red", "green", "blue", "yellow"}
    assert len(probs) == NUM_CLASSES
    assert probs[0] == 0.01
    assert len(preds) == NUM_CLASSES
    assert sum(preds) == 2
    assert preds[0] == 1
    assert preds[NUM_CLASSES - 1] == 1
