from contextlib import nullcontext
from inspect import signature
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Union

from config import (
    MLFLOW_CHECKPOINT_ARTIFACT_DIR,
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ARTIFACT_PATH,
    MLFLOW_RUN_ID,
    MLFLOW_TRACKING_URI,
)


def is_enabled(enabled: Optional[bool] = None) -> bool:
    return MLFLOW_ENABLED if enabled is None else enabled


def start_run(
    run_name: Optional[str] = None,
    enabled: Optional[bool] = None,
    tags: Optional[Dict[str, Any]] = None,
):
    if not is_enabled(enabled):
        return nullcontext()
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return mlflow.start_run(run_name=run_name, tags=tags)


def set_tags(tags: Dict[str, Any], enabled: Optional[bool] = None) -> None:
    if is_enabled(enabled):
        import mlflow

        mlflow.set_tags(tags)


def log_params(params: Dict[str, Any], enabled: Optional[bool] = None) -> None:
    if is_enabled(enabled):
        import mlflow

        mlflow.log_params(params)


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> None:
    if is_enabled(enabled):
        import mlflow

        mlflow.log_metrics(metrics, step=step)


def log_artifact(
    path: Union[str, Path],
    artifact_path: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> None:
    if is_enabled(enabled):
        import mlflow

        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_json(
    payload: Dict[str, Any],
    artifact_path: str,
    enabled: Optional[bool] = None,
) -> None:
    if not is_enabled(enabled):
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path = Path(handle.name)
    try:
        log_artifact(temp_path, artifact_path=artifact_path, enabled=enabled)
    finally:
        temp_path.unlink(missing_ok=True)


def log_model(
    model,
    artifact_path: str = MLFLOW_MODEL_ARTIFACT_PATH,
    enabled: Optional[bool] = None,
    input_example=None,
    serialization_format: Optional[str] = "pickle",
) -> None:
    if is_enabled(enabled):
        import mlflow

        parameters = signature(mlflow.pytorch.log_model).parameters
        kwargs = (
            {"name": artifact_path}
            if "name" in parameters
            else {"artifact_path": artifact_path}
        )
        if input_example is not None and "input_example" in parameters:
            kwargs["input_example"] = input_example
        if serialization_format and "serialization_format" in parameters:
            kwargs["serialization_format"] = serialization_format
        mlflow.pytorch.log_model(model, **kwargs)


def get_run(run_id: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).get_run(run_id)


def download_artifact(
    run_id: str,
    artifact_path: str,
    dst_path: Optional[Union[str, Path]] = None,
) -> str:
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).download_artifacts(
        run_id=run_id,
        path=artifact_path,
        dst_path=str(dst_path) if dst_path else None,
    )


def load_checkpoint_artifact(
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    dst_path: Optional[Union[str, Path]] = None,
):
    from cnn_model import load_model

    run_id = run_id or MLFLOW_RUN_ID
    if not run_id:
        raise ValueError("A run_id is required to load an MLflow artifact.")
    artifact_path = artifact_path or f"{MLFLOW_CHECKPOINT_ARTIFACT_DIR}/protein_model.pth"
    try:
        local_path = Path(download_artifact(run_id, artifact_path, dst_path))
    except Exception:
        artifact_dir = artifact_path.rsplit("/", 1)[0] if "/" in artifact_path else ""
        local_path = Path(download_artifact(run_id, artifact_dir, dst_path))
    if local_path.is_dir():
        candidates = sorted(local_path.rglob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found under MLflow artifact '{artifact_path}'.")
        local_path = candidates[0]
    return load_model(local_path)


def load_model_artifact(
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
):
    import mlflow

    run_id = run_id or MLFLOW_RUN_ID
    if not run_id:
        raise ValueError("A run_id is required to load an MLflow artifact.")
    artifact_path = artifact_path or MLFLOW_MODEL_ARTIFACT_PATH
    try:
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/{artifact_path}")
        model.eval()
        return model
    except Exception:
        return load_checkpoint_artifact(run_id)


def load_inference_model(run_id: Optional[str] = None):
    if run_id or MLFLOW_RUN_ID:
        return load_model_artifact(run_id or MLFLOW_RUN_ID)
    from cnn_model import load_model

    return load_model()
