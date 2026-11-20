# Step 3 — Docker, CI/CD and MLOps Integration Report

## Objective

Verify and improve containerization, Docker Compose, GitHub Actions CI/CD, GHCR publishing, MLOps integration, reproducibility, tests and documentation without changing the model architecture, class labels, preprocessing contract, original checkpoint or existing functionality.

## Changes implemented

### Containerization

The existing `Dockerfile` was improved into a multi-target build rather than being blindly recreated:

- `base`: installs `requirements.txt`, copies shared modules, both applications and `model/protein_model.pth`, creates a non-root `appuser`.
- `api`: FastAPI on port `8000` with `PROTEIN_DEVICE=cpu` and `PROTEIN_MODEL_PATH=/app/model/protein_model.pth`.
- `streamlit`: Streamlit on port `8501` with `PROTEIN_API_URL=http://fastapi:8000`.
- `mlflow`: optional MLflow UI on port `5000` with SQLite under `/app/mlflow_data`.

`requirements.txt` now explicitly includes `requests>=2.31,<3` for the Streamlit HTTP client. `.dockerignore` additionally excludes `mlflow.db`, `mlartifacts`, `scripts`, `docker-compose*.yml` and Markdown files while continuing to exclude the dataset, reports and test artifacts. The trained checkpoint is still included.

### Docker Compose

Added `docker-compose.yml` with:

- `fastapi`: API service, health check on `/health`, port `8000`.
- `streamlit`: frontend service, health check on `/_stcore/health`, port `8501`, `depends_on` FastAPI health, `PROTEIN_API_URL=http://fastapi:8000`.
- `mlflow`: service under the `mlflow` profile, port `5000`, SQLite and artifacts in the `mlflow-data` volume.

Container-to-container communication uses the Docker service hostname `fastapi`, not `localhost`.

### Streamlit-to-FastAPI client

Added `api_client.py` with:

- `check_api_health(api_url)`.
- `predict_via_api(api_url, channel_files, threshold)`.
- Response mapping from FastAPI's label-keyed probabilities and class indices into the shared `CLASS_LABELS` order.

Updated `streamlit_app.py`:

- Reads `PROTEIN_API_URL` from `config.py`.
- Uses FastAPI when `PROTEIN_API_URL` is set; otherwise preserves local checkpoint inference.
- Checks FastAPI health at startup and displays the active backend in the sidebar.
- Passes the original uploaded files to `/predict`.
- Resets upload pointers before local preview decoding.
- Converts API results to NumPy arrays before rendering the existing charts and metrics.

### CI/CD

Updated `.github/workflows/ci.yml`:

- Triggers only on pushes and pull requests to `main`.
- Test job installs `requirements-mlops.txt`, compiles all Python sources, validates preprocessing/model loading/MLflow-aware loading, tests FastAPI endpoints and runs a synthetic MLflow smoke experiment.
- Docker build job depends on the test job, validates Compose configuration, and builds `api`, `streamlit` and `mlflow` targets.
- No dataset download or deployment is performed.

Added `.github/workflows/publish.yml`:

- Separate from PR testing.
- Triggers on `v*` tags or manual dispatch.
- Publishes the `api` image to GHCR using `GITHUB_TOKEN` and semantic-version tags with `docker/metadata-action`.

### Reproducibility

Updated `human-protein-classification.py`:

- Added `set_seed()` for Python `random`, NumPy, `torch.manual_seed`, CUDA (when available), deterministic cuDNN and disabled cuDNN benchmark.
- Applied seeding inside `run_training()`.
- Added a seeded generator to the shuffled training `DataLoader`.
- Logged the seed and cuDNN settings in MLflow parameters/environment artifacts.

## Tests added

- `tests/test_seeding.py`: verifies repeated seeds reproduce Python, NumPy and Torch random streams and set deterministic cuDNN flags.
- `tests/test_api_client.py`: verifies the FastAPI health URL uses the service hostname and that API responses map into the shared 28-class order.

## Verification results

| Check | Result |
|-------|--------|
| `python -m pytest -q` | `23 passed, 1 warning in 29.29s` |
| `python -m compileall ...` | `compileall OK` |
| YAML validation (`ci.yml`, `publish.yml`, `docker-compose.yml`) | `YAML validation OK` |
| `docker compose config --quiet` | `docker compose config OK` |
| `git diff --check` | Exit `0`; only CRLF normalization warnings |
| Checkpoint SHA-256 | `1C5C06E0A1991523BA07AD29A8B7270E268DE0B787A786CE4BB04A5A40284C61` |
| Streamlit bare import | `STREAMLIT_IMPORT True local checkpoint` |
| Live HTTP client against FastAPI | `LIVE_API_CLIENT 28 2 0.7477` |
| FastAPI with original checkpoint | `ORIGINAL_CHECKPOINT_LOAD 200 200 protein_model.pth` |
| FastAPI with MLflow artifact | `MLFLOW_ARTIFACT_API 200 200` |
| MLflow old artifacts still present | Both Step 2 runs retain checkpoint, model and config artifacts |
| New MLflow smoke run | `6cfbb0c16b8943d5a40e5c48e6a82eda` with checkpoint and model artifacts |

The MLflow store at `C:\Users\LENOVO\AppData\Local\Temp\opencode\hpa-mlflow-smoke2` still contains the Step 2 runs:

- `439572e575a940f49307e6711a6ee01d`
- `fb2928b324144edb81efc2845cb17118`

A new Step 3 smoke run was written as `6cfbb0c16b8943d5a40e5c48e6a82eda`; existing run IDs and artifacts were not overwritten.

## Docker limitation

The Docker CLI is installed (`Docker version 28.4.0`), but the daemon is not running. `docker build --check .` fails with:

```text
ERROR: error during connect: Head "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

Therefore local image builds and container startup were not executed. `docker compose config --quiet` succeeded because it validates configuration without contacting the daemon. GitHub Actions CI now performs the real multi-target builds on runners with Docker available.

## Documentation

`README.md` was updated with:

- Local and Docker architecture diagrams.
- MLOps workflow diagram.
- Streamlit/FastAPI backend selection.
- Multi-target Docker instructions and Compose commands.
- Optional MLflow profile instructions.
- CI/CD and GHCR publishing explanation.
- Reproducibility and verification status.

## Scope compliance

- CNN architecture, 4-channel input, 28 labels and threshold `0.3` unchanged.
- Preprocessing contract unchanged.
- Original checkpoint unchanged and verified by SHA-256.
- No dataset downloaded.
- No deployment triggered.
- No commit or push performed.
