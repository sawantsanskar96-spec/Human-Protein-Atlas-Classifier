# Step 2 — MLflow Implementation

Date: 2026-11-20  
Repository: `D:\Human Protein Classification\Human-Protein-Atlas-Classifier`

## 1. Scope

This step completed and verified the optional MLflow foundation created in Step 1. The Step 1 report and the existing `tracking.py`, `requirements-mlops.txt`, training script and FastAPI loader were inspected before making changes.

No full training run was performed. The real dataset is not present in this repository, so verification used a lightweight synthetic-data experiment only.

The original checkpoint was not modified:

```text
SHA-256: 1C5C06E0A1991523BA07AD29A8B7270E268DE0B787A786CE4BB04A5A40284C61
```

No commit or push was performed.

## 2. What was implemented

### 2.1 Installation

Installed the existing optional dependency manifest:

```bash
python -m pip install -r requirements-mlops.txt
```

Installed MLflow version: `3.18.0`.

`requirements-mlops.txt` and `tracking.py` were not recreated; they were installed/updated in place.

### 2.2 `tracking.py` extensions

The existing wrapper was extended with:

- `start_run(run_name=..., tags=...)` so every execution creates a separately named MLflow run.
- `set_tags()` for project/task/model metadata.
- `log_artifact(..., artifact_path=...)` to place the best checkpoint under `checkpoints/`.
- `log_json()` to store model configuration and environment information as JSON artifacts.
- `log_model()` compatibility for MLflow 2.x and 3.x, using `pickle` serialization to avoid MLflow 3's PT2 `input_example` requirement.
- `get_run()`, `download_artifact()`, `load_checkpoint_artifact()`, `load_model_artifact()` and `load_inference_model()` for artifact retrieval and inference.
- `load_inference_model()` falls back to the local committed checkpoint when no MLflow run ID is configured.

MLflow remains imported only inside functions that need it. Normal local development, pytest and FastAPI still work with `requirements.txt` alone.

### 2.3 Training integration

`human-protein-classification.py` was refactor-extended without changing the CNN architecture:

- `run_training(...)` is now reusable and owns the full MLflow-instrumented train/validate loop.
- `main()` remains the real training entry point: `python human-protein-classification.py`.
- `run_training()` was called by the synthetic smoke script, so the smoke experiment uses the same tracking code as real training.
- Each execution creates a new run named from `MLFLOW_RUN_NAME` or a timestamped default.
- Best-checkpoint saving still writes to `PROTEIN_CHECKPOINT_OUTPUT_PATH` (default `artifacts/checkpoints/protein_model.pth`), never to the committed model.

### 2.4 Tracking backend

MLflow 3.x refuses the old `file:./mlruns` default unless an opt-out flag is set. The default was changed to a database backend:

```text
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

`.gitignore` was updated to ignore `mlflow.db` and `mlartifacts/`.

### 2.5 Synthetic smoke experiment

Added `scripts/mlflow_smoke_test.py`. It:

- Generates tiny synthetic 4-channel 128 x 128 samples in memory.
- Uses the same `CustomCNN`, preprocessing helpers and `run_training()` as real training.
- Runs 1 epoch by default with 8 train samples and 2 validation samples.
- Writes checkpoints only to a temporary directory.
- Supports `--runs`, `--epochs`, `--batch-size`, `--samples`, `--tracking-uri` and `--experiment`.
- Does not download the dataset or touch the original checkpoint.

### 2.6 FastAPI compatibility

`app.py` now calls `tracking.load_inference_model(MLFLOW_RUN_ID)`.

- With `PROTEIN_MLFLOW_RUN_ID` unset: uses `model/protein_model.pth` exactly as before.
- With `PROTEIN_MLFLOW_RUN_ID` set: downloads/loads the selected MLflow model artifact.
- The model-loading print statement no longer emits a non-ASCII checkmark, avoiding `UnicodeEncodeError` under `cp1252` consoles.

## 3. How training integrates with MLflow

Training flow with `MLFLOW_ENABLED=1`:

1. `main()` builds the real dataset loaders.
2. `run_training()` starts one MLflow run with a unique name and metadata tags.
3. Hyperparameters and environment values are logged as run parameters.
4. Model configuration and environment details are logged as JSON artifacts.
5. Each epoch logs training loss, validation loss, macro F1 and micro F1 at `step=epoch`.
6. The best checkpoint is saved locally and logged once as an artifact under `checkpoints/`.
7. The best checkpoint is loaded and logged as an MLflow PyTorch model named `model`.
8. Best macro F1 and best epoch are logged as final metrics.
9. The run ID is printed.

With `MLFLOW_ENABLED=0` (default), all tracking calls are no-ops and training proceeds without MLflow.

## 4. Parameters and metrics tracked

Required parameters:

| Parameter | Source |
|-----------|--------|
| `learning_rate` | `LEARNING_RATE` / `run_training(learning_rate=...)` |
| `batch_size` | `BATCH_SIZE` / `run_training(batch_size=...)` |
| `epochs` | `EPOCHS` / `run_training(epochs=...)` |
| `optimizer` | `Adam` |
| `loss_function` | `BCEWithLogitsLoss` |
| `image_size` | `IMG_SIZE` |
| `input_channels` | `INPUT_CHANNELS` (4) |
| `num_classes` | `NUM_CLASSES` (28) |
| `prediction_threshold` | `PREDICTION_THRESHOLD` (0.3) |
| `random_seed` | `RANDOM_STATE` (42) |

Additional parameters: validation batch size, validation split, device, architecture, `img_size`, `random_state`, `loss`, and `env_*` values for Python, PyTorch, NumPy, pandas, scikit-learn, platform, CPU/GPU and git commit.

Per-epoch metrics:

| Metric | Meaning |
|--------|---------|
| `train_loss` / `training_loss` | Mean training loss for the epoch |
| `val_loss` / `validation_loss` | Mean validation loss for the epoch |
| `macro_f1` | Macro F1 at the configured threshold |
| `micro_f1` | Micro F1 at the configured threshold |

Final metrics: `best_macro_f1`, `best_epoch`.

Aliases were kept for Step 1 comparison compatibility.

## 5. Artifact management

Each run logs:

- `config/model_config.json` — architecture, input shape, class labels, threshold, optimizer, loss, batch sizes and seed.
- `config/environment.json` — Python/PyTorch/library versions, device, platform and git commit.
- `checkpoints/<checkpoint-filename>` — the best state-dict checkpoint.
- An MLflow PyTorch model logged as `model`, loadable via `runs:/<run-id>/model`.

The committed checkpoint at `model/protein_model.pth` is untouched.

## 6. Launch MLflow UI

From the repository root:

```powershell
python -m pip install -r requirements-mlops.txt
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://localhost:5000`.

## 7. Compare experiments

In the MLflow UI:

1. Open the `human-protein-atlas-classifier` experiment (or the smoke experiment name used during testing).
2. Select multiple runs with the checkboxes.
3. Choose **Compare**.
4. Compare `learning_rate`, `batch_size`, `epochs`, `macro_f1`, `micro_f1`, `train_loss` and `validation_loss`.

Programmatic comparison:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
experiment = client.get_experiment_by_name("human-protein-atlas-classifier")
runs = client.search_runs([experiment.experiment_id])
```

## 8. Load model artifacts

FastAPI selection by run ID:

```powershell
$env:PROTEIN_MLFLOW_RUN_ID = "<run-id>"
uvicorn app:app --reload
```

Python helpers:

```python
import tracking

model = tracking.load_model_artifact("<run-id>")
checkpoint_model = tracking.load_checkpoint_artifact("<run-id>")
```

`load_model_artifact()` first tries `mlflow.pytorch.load_model("runs:/<run-id>/model")`. If that fails, it falls back to downloading the logged state-dict checkpoint and loading it through `cnn_model.load_model()`.

## 9. Actual test results

### 9.1 Synthetic smoke experiment

Command run from a temporary directory so the SQLite store and artifacts stayed outside the repository:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
python "D:\Human Protein Classification\Human-Protein-Atlas-Classifier\scripts\mlflow_smoke_test.py" --runs 2 --epochs 1 --samples 8 --batch-size 2 --experiment smoke-step2
```

Two separate runs were created:

| Run ID | Status | train_loss | validation_loss | macro_f1 | micro_f1 |
|--------|--------|-----------|-----------------|----------|----------|
| `439572e575a940f49307e6711a6ee01d` | FINISHED | 0.63418 | 0.66961 | 0.09524 | 0.13333 |
| `fb2928b324144edb81efc2845cb17118` | FINISHED | 0.64703 | 0.67672 | 0.09524 | 0.13333 |

These numbers come from random synthetic data and are **not** model performance metrics.

Verification via `MlflowClient`:

```text
EXPERIMENT smoke-step2 1
RUN_COUNT 2
MISSING_PARAMS []                 (both runs)
METRIC_STEPS train_loss=[1], validation_loss=[1], macro_f1=[1], micro_f1=[1]
ARTIFACTS ['checkpoints', 'config']
CONFIG_ARTIFACTS ['config/environment.json', 'config/model_config.json']
CHECKPOINT_ARTIFACTS ['checkpoints/smoke_run_1.pth'] / ['checkpoints/smoke_run_2.pth']
```

### 9.2 Artifact loading

```text
LOADED_MODEL_OUTPUT (1, 28) training=False
LOADED_CHECKPOINT_MODEL CustomCNN training=False
```

### 9.3 FastAPI with selected MLflow run

With `PROTEIN_MLFLOW_RUN_ID=439572e575a940f49307e6711a6ee01d`:

```text
Model loaded from MLflow run 439572e575a940f49307e6711a6ee01d
HEALTH 200 {'status': 'ok', 'device': 'cpu'}
PREDICT 200 28
THRESHOLD 0.3
```

The existing four-upload `/predict` endpoint worked unchanged.

### 9.4 Pytest

```bash
python -m pytest -q
```

Result:

```text
20 passed, 1 warning in 17.24s
```

Three tests were added for optional tracking behavior. The single warning came from `pytest-asyncio`/`anyio` plugin configuration and is unrelated to project code.

### 9.5 Syntax and diff checks

```text
py_compile OK
git diff --check OK
```

## 10. Remaining limitations

1. No full training was run because the dataset is unavailable and permission was not granted.
2. Smoke metrics are synthetic and must not be reported as real model performance.
3. MLflow 3 logs PyTorch models through its logged-model mechanism; `client.list_artifacts(run_id)` may not list the `model` directory even though `runs:/<run-id>/model` loads successfully.
4. MLflow emitted a warning that pickle/CloudPickle serialization can execute arbitrary code during deserialization. Only load artifacts from trusted tracking stores.
5. Model registry stages/aliases and promotion approvals were not implemented; this step covers experiment tracking, artifact logging and artifact-based inference.
6. CI does not yet run the MLflow smoke script; it remains a local verification utility.
7. Loading an MLflow PyTorch artifact requires a compatible Python/PyTorch environment and the repository modules on `PYTHONPATH`.
