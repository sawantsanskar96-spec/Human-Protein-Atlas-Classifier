# Step 1 — Repository Audit and MLOps Foundation

Date: 2026-11-20  
Repository: `D:\Human Protein Classification\Human-Protein-Atlas-Classifier`

## 1. Scope

The entire repository was inspected before changes:

- All tracked files: `app.py`, `streamlit_app.py`, `human-protein-classification.py`, `model/protein_model.pth`, `README.md`
- Git history: 2 commits (`b59d6e8 Initial commit`, `d9104ce Updated README`)
- Hidden files: only `.git/` metadata was present; no `.gitignore`, no GitHub Actions workflow
- Python files: 3 root-level scripts
- Notebooks: none
- Model/checkpoint files: one file, `model/protein_model.pth` (1,601,505 bytes)
- Dataset: not present in the repository; not downloaded during this step
- Dependency manifests: none present before this step
- Tests: none present before this step

No commit, push, retraining, dataset download or checkpoint overwrite was performed.

Checkpoint SHA-256 before and after all changes:

```text
1C5C06E0A1991523BA07AD29A8B7270E268DE0B787A786CE4BB04A5A40284C61
```

## 2. Existing architecture

### 2.1 Model

The actual model definition was duplicated in three files before the changes:

- `human-protein-classification.py:66-101`
- `app.py:30-45`
- `streamlit_app.py:219-233`

All three copies described the same `CustomCNN`:

```text
Conv2d(4, 32, k=3, p=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2)
Conv2d(32, 64, k=3, p=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2)
Conv2d(64, 128, k=3, p=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2d(2)
Conv2d(128, 256, k=3, p=1) -> BatchNorm2d(256) -> ReLU -> AdaptiveAvgPool2d((1, 1))
Flatten -> Dropout(0.4) -> Linear(256, 28)
```

The model outputs 28 raw logits. `BCEWithLogitsLoss` is used in training; sigmoid is applied explicitly for validation and inference. The prediction threshold is `0.3`.

### 2.2 Data and preprocessing

Training used `dataset/train.csv` and `dataset/train/` with four grayscale PNGs per sample (`_red`, `_green`, `_blue`, `_yellow`). Each channel was resized to 128 x 128, stacked in RGBY order, scaled by `/255.0`, and converted to a `torch.float32` tensor. Labels were parsed from whitespace-separated class indices into a 28-element multi-hot vector.

Inference decoded four uploaded images from bytes with `cv2.imdecode(..., cv2.IMREAD_GRAYSCALE)`, resized to 128 x 128, stacked in RGBY order, scaled by `/255.0`, added a batch dimension, and ran the same CNN.

### 2.3 Application layers

- FastAPI (`app.py`): import-time checkpoint load, CORS middleware, endpoints `/`, `/health`, `/classes`, `/predict`.
- Streamlit (`streamlit_app.py`): cached model load, four-channel upload UI, previews, threshold slider, probability chart and top-5 results.
- Training (`human-protein-classification.py`): top-level script with an 80/20 split, Adam optimizer, 25 epochs, batch size 16, macro/micro F1 validation, and best-checkpoint saving.

## 3. Existing execution flow

### 3.1 Training flow

1. Import time immediately read `dataset/train.csv`.
2. Converted `Target` strings to multi-hot vectors.
3. Split into train/validation with `test_size=0.2`, `random_state=42`.
4. Build `ProteinDataset` and `DataLoader`s.
5. Instantiate `CustomCNN`, `BCEWithLogitsLoss`, and Adam (`lr=0.001`).
6. For 25 epochs: train, validate with sigmoid + `0.3` threshold, print loss/macro F1/micro F1.
7. Save `model.state_dict()` to `backend/model/protein_model.pth` whenever macro F1 improved.

Because all of this ran at module import, importing the training script from CI or a test runner would have attempted to read the dataset and start training.

### 3.2 FastAPI flow

1. At import, instantiate `CustomCNN` and load `backend/model/protein_model.pth`.
2. `/predict` reads four uploaded files, decodes each to grayscale 128 x 128, stacks RGBY, scales to `[0, 1]`, runs the model, applies sigmoid, thresholds at the supplied value (default `0.3`), and returns class labels, indices and all 28 probabilities.

### 3.3 Streamlit flow

1. Page config and CSS load.
2. Cached `load_model()` loads `backend/model/protein_model.pth`.
3. User uploads four channels; previews are rendered.
4. `Run Classification` stacks channels, runs inference, thresholds and displays results.

## 4. Repository structure

### 4.1 Before

```text
.
├── app.py
├── human-protein-classification.py
├── model/
│   └── protein_model.pth
├── streamlit_app.py
└── README.md
```

### 4.2 After

```text
.
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── Dockerfile
├── README.md
├── app.py
├── cnn_model.py
├── config.py
├── human-protein-classification.py
├── model/
│   └── protein_model.pth          # unchanged
├── preprocessing.py
├── reports/
│   └── STEP_1_REPOSITORY_AUDIT.md
├── requirements-mlops.txt
├── requirements.txt
├── streamlit_app.py
├── tests/
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_model.py
│   └── test_preprocessing.py
└── tracking.py
```

## 5. Training and inference compatibility findings

### 5.1 Architecture match

There was **no architecture mismatch**. The training, FastAPI and Streamlit CNN definitions were structurally identical before the refactor.

The checkpoint was inspected directly. Its state dict is an `OrderedDict` whose keys and shapes exactly match `CustomCNN`:

```text
features.0.weight  (32, 4, 3, 3)
features.4.weight  (64, 32, 3, 3)
features.8.weight  (128, 64, 3, 3)
features.12.weight (256, 128, 3, 3)
classifier.2.weight (28, 256)
classifier.2.bias   (28,)
```

Full key sequence:

```text
features.0.weight, features.0.bias,
features.1.weight, features.1.bias, features.1.running_mean,
features.1.running_var, features.1.num_batches_tracked,
features.4.weight, features.4.bias,
features.5.weight, features.5.bias, features.5.running_mean,
features.5.running_var, features.5.num_batches_tracked,
features.8.weight, features.8.bias,
features.9.weight, features.9.bias, features.9.running_mean,
features.9.running_var, features.9.num_batches_tracked,
features.12.weight, features.12.bias,
features.13.weight, features.13.bias, features.13.running_mean,
features.13.running_var, features.13.num_batches_tracked,
classifier.2.weight, classifier.2.bias
```

`model.load_state_dict(state_dict)` succeeds against the shared definition. The architecture was therefore extracted verbatim into `cnn_model.py` rather than redesigned.

### 5.2 Checkpoint path mismatch (real bug found)

The trained file exists at:

```text
model/protein_model.pth
```

But all three original scripts referenced:

```text
backend/model/protein_model.pth
```

Consequences:

- `import app` failed with `RuntimeError: Model file not found at 'backend/model/protein_model.pth'`.
- `streamlit_app.py` silently entered its "Model not found" UI state.
- Training would have tried to save into a non-existent `backend/model/` directory.

Resolution:

- Default inference path is now the real file: `PROTEIN_MODEL_PATH` → `model/protein_model.pth`.
- `backend/model/protein_model.pth` remains available as a legacy fallback via `PROTEIN_LEGACY_MODEL_PATH`.
- Training output now defaults to `artifacts/checkpoints/protein_model.pth` so future training cannot overwrite the committed trained checkpoint.
- All paths can be overridden with environment variables.

### 5.3 Preprocessing comparison

| Aspect | Original training | Original inference | Compatibility |
|---|---|---|---|
| Channel source | `cv2.imread` from disk | `cv2.imdecode` from uploaded bytes | Equivalent content, different source |
| Colour mode | `IMREAD_GRAYSCALE` | `IMREAD_GRAYSCALE` | Match |
| Resize | `(128, 128)` | `(128, 128)` | Match |
| Channel order | red, green, blue, yellow | red, green, blue, yellow | Match |
| Scaling | `image / 255.0` then `torch.tensor(..., float32)` | `astype(np.float32) / 255.0` | Numerically equivalent |
| Batch dimension | Added by `DataLoader` | Added with `unsqueeze(0)` | Match |
| Threshold | `0.3` | `0.3` default | Match |
| Validation of 4 channels/shape | Implicit | None originally | Added safely in shared preprocessing |

Inference now uses the same `stack_channels` helper that training uses, with explicit validation of four channels and 128 x 128 shape.

## 6. Dependencies and Python versions

Before this step there was no `requirements.txt` and no dependency pinning.

Observed local environment:

| Package | Version |
|---|---|
| Python | 3.11.6 |
| torch | 2.13.0 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.0 |
| fastapi | 0.115.12 |
| uvicorn | 0.34.0 |
| pydantic | 2.11.3 |
| streamlit | 1.44.1 |
| matplotlib | 3.11.0 |
| pytest | 9.2.0 |
| MLflow | not installed |
| OpenCV (`cv2`) | initially missing; installed `opencv-python-headless 4.14.0.94` during the audit |
| Docker CLI | 28.4.0 (daemon not running locally) |

Added manifests:

- `requirements.txt`: core runtime plus `pytest` and `httpx` for API tests.
- `requirements-mlops.txt`: `requirements.txt` plus `mlflow>=2.9,<4`.

MLflow is imported only inside `tracking.py` functions and only when `MLFLOW_ENABLED` is true, so local development and tests work with just `requirements.txt`.

## 7. Hardcoded paths and configuration values found

Original hardcoded values:

- `dataset/train.csv`
- `dataset/train`
- `backend/model/protein_model.pth`
- `IMG_SIZE = 128` in three files
- `NUM_CLASSES = 28` in three files
- Prediction threshold `0.3` in training, FastAPI and Streamlit
- `CLASS_LABELS` duplicated in FastAPI and Streamlit
- Epochs `25`, batch size `16`, validation batch size `64`, validation split `0.2`, random state `42`, learning rate `0.001`
- CORS `allow_origins=["*"]`
- Uvicorn host `0.0.0.0`, port `8000`, reload `True`

These are now centralized in `config.py` with environment overrides. CORS remains permissive to preserve existing behavior.

## 8. README inconsistencies found

The original `README.md` was UTF-16 LE with a BOM and contained corrupted emoji/control characters (for example `?#`, `??` and control-byte artifacts). It also documented a structure that did not exist:

| README claimed | Actual before changes |
|---|---|
| `data/` | no data directory in repo |
| `model/model.py` | no such file |
| `model/train.py` | no such file |
| `frontend.py` | `streamlit_app.py` |
| `utils.py` | no such file |
| `requirements.txt` | absent |
| `pip install -r requirements.txt` | would fail |
| `streamlit run frontend.py` | would fail |
| "Sigmoid activation" as a layer | logits + explicit sigmoid; `BCEWithLogitsLoss` |

The reported F1 values were retained as inherited documentation/UI values, but the README now explicitly states they were not re-evaluated in this step. No performance metric was invented.

## 9. Changes made

### 9.1 Minimal reorganization

- Added `cnn_model.py`: single source of truth for `CustomCNN` and checkpoint loading.
- Added `preprocessing.py`: shared decoding, resizing, channel stacking, prediction and label helpers.
- Added `config.py`: paths, constants, class labels and environment settings.
- Added `tracking.py`: optional MLflow wrapper with no hard dependency.

### 9.2 Entry points preserved

- `human-protein-classification.py` remains the training entry point, but training is now inside `main()` and guarded by `if __name__ == "__main__":`. Hyperparameters, loss, threshold, split and architecture are unchanged.
- `app.py` remains the FastAPI entry point with the same endpoints and response schema.
- `streamlit_app.py` remains the Streamlit entry point with the same UI behavior.

### 9.3 Safety and MLOps foundation

- Fixed inference checkpoint path without touching the existing checkpoint.
- Prevented future training from overwriting the committed checkpoint by default.
- Added `pytest` structure with 17 tests.
- Added environment variable configuration.
- Added optional MLflow tracking for parameters, metrics, artifacts and best model.
- Added `Dockerfile` and `.dockerignore`.
- Added `.github/workflows/ci.yml` for pytest and Docker build checks.
- Added `.gitignore`.
- Rewrote `README.md` as UTF-8 with accurate paths and commands.

## 10. Tests performed and results

### 10.1 Pytest

Command:

```bash
python -m pytest -q
```

Result:

```text
17 passed, 1 warning in 14.73s
```

Coverage:

- `tests/test_config.py`: class labels, default hyperparameters, checkpoint existence, environment override
- `tests/test_model.py`: checkpoint state-dict key/shape compatibility, eval mode, 28-logit output
- `tests/test_preprocessing.py`: PNG decoding, invalid input handling, image loading, RGBY normalization, multi-hot labels
- `tests/test_api.py`: `/`, `/health`, `/classes`, `/predict`, invalid threshold, invalid image

The single warning came from `pytest-asyncio`/`anyio` plugin configuration in the local environment and is unrelated to project code.

### 10.2 Syntax check

Command:

```bash
python -m py_compile app.py streamlit_app.py human-protein-classification.py config.py cnn_model.py preprocessing.py tracking.py tests/*.py
```

Result: passed.

### 10.3 Inference verification

- FastAPI `/predict` was exercised through `TestClient` with four synthetic PNG channels; the response contained exactly 28 probabilities and valid class indices.
- `state_dict` loading into the shared model succeeded.
- `import app` now succeeds locally (previously failed).
- `import streamlit_app` succeeded with `model_ok=True`. Streamlit emitted expected bare-mode warnings because it was imported outside a Streamlit server; no application error occurred.

### 10.4 Checkpoint integrity

SHA-256 before and after changes is identical:

```text
1C5C06E0A1991523BA07AD29A8B7270E268DE0B787A786CE4BB04A5A40284C61
```

### 10.5 Docker verification

`docker build --check .` could not run because the Docker CLI is installed but the Docker Desktop daemon is not running in this environment. The Dockerfile and CI build step were added but not built locally.

### 10.6 MLflow verification

MLflow is not installed locally. This is expected: `MLFLOW_ENABLED` defaults to `0`, and the test suite passed without MLflow present.

## 11. Remaining issues

1. Dataset is absent, so training, retraining and metric reproduction were not attempted.
2. Existing F1 values are inherited from the prior README/Streamlit UI and were not independently verified.
3. Docker image build was not executed locally because the Docker daemon was unavailable; GitHub Actions will validate it on push/PR.
4. MLflow tracking was not exercised end-to-end because MLflow is not installed locally. The optional dependency file and wrapper are in place.
5. CORS remains `allow_origins=["*"]`, preserved from the original project. It should be narrowed before production deployment.
6. Streamlit still uses an external Google Fonts import in CSS; offline deployments may fall back to system fonts.
7. GitHub Actions currently provides CI and Docker build validation only; no deployment/CD target was configured because none was specified.
