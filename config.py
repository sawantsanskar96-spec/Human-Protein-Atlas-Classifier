import os
from pathlib import Path
from typing import List, Union

import torch

ROOT_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


CLASS_LABELS: List[str] = [
    "Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center",
    "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus",
    "Peroxisomes", "Endosomes", "Lysosomes", "Intermediate filaments",
    "Actin filaments", "Focal adhesion sites", "Microtubules", "Microtubule ends",
    "Cytokinetic bridge", "Mitotic spindle", "Microtubule organizing center",
    "Centrosome", "Lipid droplets", "Plasma membrane", "Cell junctions",
    "Mitochondria", "Aggresome", "Cytosol", "Cytoplasmic bodies", "Rods & rings",
]

IMG_SIZE = int(os.getenv("PROTEIN_IMG_SIZE", "128"))
NUM_CLASSES = int(os.getenv("PROTEIN_NUM_CLASSES", "28"))
PREDICTION_THRESHOLD = float(os.getenv("PROTEIN_PREDICTION_THRESHOLD", "0.3"))

DEVICE = torch.device(
    os.getenv("PROTEIN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
)

MODEL_PATH = _env_path("PROTEIN_MODEL_PATH", "model/protein_model.pth")
LEGACY_MODEL_PATH = _env_path("PROTEIN_LEGACY_MODEL_PATH", "backend/model/protein_model.pth")
CHECKPOINT_OUTPUT_PATH = _env_path(
    "PROTEIN_CHECKPOINT_OUTPUT_PATH", "artifacts/checkpoints/protein_model.pth"
)

DATASET_CSV_PATH = _env_path("PROTEIN_DATASET_CSV", "dataset/train.csv")
DATASET_IMAGE_DIR = _env_path("PROTEIN_DATASET_IMAGE_DIR", "dataset/train")

BATCH_SIZE = int(os.getenv("PROTEIN_BATCH_SIZE", "16"))
VALIDATION_BATCH_SIZE = int(os.getenv("PROTEIN_VALIDATION_BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("PROTEIN_EPOCHS", "25"))
LEARNING_RATE = float(os.getenv("PROTEIN_LEARNING_RATE", "0.001"))
VALIDATION_SIZE = float(os.getenv("PROTEIN_VALIDATION_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("PROTEIN_RANDOM_STATE", "42"))

API_HOST = os.getenv("PROTEIN_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PROTEIN_API_PORT", "8000"))
API_RELOAD = os.getenv("PROTEIN_API_RELOAD", "1").lower() in {"1", "true", "yes", "on"}
API_URL = os.getenv("PROTEIN_API_URL", "").rstrip("/")

INPUT_CHANNELS = 4

MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "human-protein-atlas-classifier"
)
MLFLOW_RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "")
MLFLOW_RUN_ID = os.getenv("PROTEIN_MLFLOW_RUN_ID", "")
MLFLOW_MODEL_ARTIFACT_PATH = os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "model")
MLFLOW_CHECKPOINT_ARTIFACT_DIR = os.getenv(
    "MLFLOW_CHECKPOINT_ARTIFACT_DIR", "checkpoints"
)
MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "human-protein-atlas-classifier"
)


def resolve_model_path(candidate: Union[str, Path, None] = None) -> Path:
    path = Path(candidate) if candidate else MODEL_PATH
    if not path.is_absolute():
        path = ROOT_DIR / path
    if path.exists():
        return path
    if candidate is None and LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    raise FileNotFoundError(
        "Model checkpoint not found at "
        f"'{path}'. Set PROTEIN_MODEL_PATH to the checkpoint location."
    )
