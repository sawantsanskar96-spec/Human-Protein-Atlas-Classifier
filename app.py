import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import tracking
from config import (
    API_HOST,
    API_PORT,
    API_RELOAD,
    CLASS_LABELS,
    DEVICE,
    IMG_SIZE,
    MLFLOW_RUN_ID,
    MODEL_PATH,
    NUM_CLASSES,
    PREDICTION_THRESHOLD,
)
from preprocessing import decode_channel_bytes, predict_from_channels

# ── Model definition (shared with training: cnn_model.CustomCNN) ──────────────
try:
    model = tracking.load_inference_model(MLFLOW_RUN_ID or None)
    source = f"MLflow run {MLFLOW_RUN_ID}" if MLFLOW_RUN_ID else MODEL_PATH
    print(f"Model loaded from {source}")
except Exception as exc:
    raise RuntimeError(f"Model could not be loaded. {exc}") from exc

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Human Protein Classifier API",
    description="Multi-label classification of human protein subcellular localisation using 4-channel fluorescence images.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded PNG/JPG bytes → grayscale numpy array (IMG_SIZE × IMG_SIZE)."""
    try:
        return decode_channel_bytes(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def predict(channels: List[np.ndarray], threshold: float = PREDICTION_THRESHOLD):
    """Stack 4 channels → tensor → model → probabilities & predictions."""
    return predict_from_channels(model, channels, threshold=threshold, device=DEVICE)

# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_classes: List[str]
    class_indices: List[int]
    probabilities: dict
    threshold_used: float

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Human Protein Classifier API is running 🧬", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.get("/classes")
def get_classes():
    """Return all 28 class labels with their indices."""
    return {i: label for i, label in enumerate(CLASS_LABELS)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    red:    UploadFile = File(..., description="Red-channel PNG"),
    green:  UploadFile = File(..., description="Green-channel PNG"),
    blue:   UploadFile = File(..., description="Blue-channel PNG"),
    yellow: UploadFile = File(..., description="Yellow-channel PNG"),
    threshold: float = PREDICTION_THRESHOLD,
):
    """
    Upload 4 single-channel fluorescence images (red, green, blue, yellow)
    and receive multi-label protein localisation predictions.
    """
    if not (0.0 < threshold < 1.0):
        raise HTTPException(status_code=422, detail="Threshold must be between 0 and 1.")

    channels = []
    for channel_file in [red, green, blue, yellow]:
        raw = await channel_file.read()
        channels.append(decode_image(raw))

    probs, preds = predict(channels, threshold=threshold)

    predicted_indices = [i for i, p in enumerate(preds) if p == 1]
    predicted_labels  = [CLASS_LABELS[i] for i in predicted_indices]
    prob_dict         = {CLASS_LABELS[i]: round(probs[i], 4) for i in range(NUM_CLASSES)}

    return PredictionResponse(
        predicted_classes=predicted_labels,
        class_indices=predicted_indices,
        probabilities=prob_dict,
        threshold_used=threshold,
    )

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
