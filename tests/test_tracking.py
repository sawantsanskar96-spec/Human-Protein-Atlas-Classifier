import tracking
from cnn_model import CustomCNN


def test_tracking_disabled_context():
    with tracking.start_run(enabled=False) as run:
        assert run is None


def test_tracking_noop_logging_when_disabled():
    tracking.set_tags({}, enabled=False)
    tracking.log_params({}, enabled=False)
    tracking.log_metrics({}, enabled=False)
    tracking.log_json({"key": "value"}, "config/test.json", enabled=False)
    tracking.log_model(object(), enabled=False)


def test_load_inference_model_without_run_id(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_RUN_ID", "")
    model = tracking.load_inference_model()
    assert isinstance(model, CustomCNN)
    assert model.training is False
