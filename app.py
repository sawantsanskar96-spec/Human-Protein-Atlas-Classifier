import io
import cv2
import torch
import numpy as np
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE   = 128
NUM_CLASSES = 28
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "backend/model/protein_model.pth"

# Human Protein Atlas — 28 class labels
CLASS_LABELS = [
    "Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center",
    "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus",
    "Peroxisomes", "Endosomes", "Lysosomes", "Intermediate filaments",
    "Actin filaments", "Focal adhesion sites", "Microtubules", "Microtubule ends",
    "Cytokinetic bridge", "Mitotic spindle", "Microtubule organizing center",
    "Centrosome", "Lipid droplets", "Plasma membrane", "Cell junctions",
    "Mitochondria", "Aggresome", "Cytosol", "Cytoplasmic bodies", "Rods & rings",
]

# ── Model definition (must match training) ────────────────────────────────────
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Load model ─────────────────────────────────────────────────────────────────
model = CustomCNN().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅  Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at '{MODEL_PATH}'. Train the model first.")

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
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid PNG/JPG.")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def predict(channels: List[np.ndarray], threshold: float = 0.3):
    """Stack 4 channels → tensor → model → probabilities & predictions."""
    image = np.stack(channels, axis=0).astype(np.float32) / 255.0
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs > threshold).astype(int)
    return probs.tolist(), preds.tolist()

# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_classes: List[str]
    class_indices: List[int]
    probabilities: dict          # label → probability
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
    threshold: float = 0.3,
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
