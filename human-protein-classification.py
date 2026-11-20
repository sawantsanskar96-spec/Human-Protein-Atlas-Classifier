from datetime import datetime
import platform
from pathlib import Path
import random
import subprocess
from typing import Optional

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import sklearn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import tracking
from cnn_model import CustomCNN, load_model as load_classification_model
from config import (
    BATCH_SIZE,
    CHECKPOINT_OUTPUT_PATH,
    CLASS_LABELS,
    DATASET_CSV_PATH,
    DATASET_IMAGE_DIR,
    DEVICE,
    EPOCHS,
    IMG_SIZE,
    INPUT_CHANNELS,
    LEARNING_RATE,
    MLFLOW_CHECKPOINT_ARTIFACT_DIR,
    MLFLOW_RUN_NAME,
    NUM_CLASSES,
    PREDICTION_THRESHOLD,
    RANDOM_STATE,
    ROOT_DIR,
    VALIDATION_BATCH_SIZE,
    VALIDATION_SIZE,
)
from preprocessing import create_multilabel, load_channel_image, stack_channels


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ProteinDataset(Dataset):
    def __init__(self, dataframe, img_dir):
        self.df = dataframe
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['Id']

        channels = [
            load_channel_image(self.img_dir / f"{img_id}_{channel}.png")
            for channel in ("red", "green", "blue", "yellow")
        ]
        image = stack_channels(channels)
        label = row['target_vec']

        return torch.from_numpy(image), torch.as_tensor(label, dtype=torch.float32)


def build_data_loaders():
    train_csv = pd.read_csv(DATASET_CSV_PATH)
    train_csv['target_vec'] = train_csv['Target'].apply(
        lambda target: create_multilabel(target, NUM_CLASSES)
    )

    train_df, valid_df = train_test_split(
        train_csv, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )

    train_dataset = ProteinDataset(train_df, DATASET_IMAGE_DIR)
    valid_dataset = ProteinDataset(valid_df, DATASET_IMAGE_DIR)

    generator = torch.Generator().manual_seed(RANDOM_STATE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)
    return train_loader, valid_loader


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _environment_info():
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
    }


def _model_config(batch_size, validation_batch_size):
    return {
        "architecture": "CustomCNN",
        "input_channels": INPUT_CHANNELS,
        "image_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "class_labels": CLASS_LABELS,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "optimizer": "Adam",
        "loss_function": "BCEWithLogitsLoss",
        "batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "random_seed": RANDOM_STATE,
    }


def run_training(
    train_loader,
    valid_loader,
    *,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    validation_batch_size: int = VALIDATION_BATCH_SIZE,
    checkpoint_path=CHECKPOINT_OUTPUT_PATH,
    run_name: Optional[str] = None,
    mlflow_enabled: Optional[bool] = None,
):
    checkpoint_path = Path(checkpoint_path)
    run_name = run_name or MLFLOW_RUN_NAME or f"cnn-{datetime.now():%Y%m%d-%H%M%S}"
    set_seed(RANDOM_STATE)

    model = CustomCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    environment = _environment_info()
    model_config = _model_config(batch_size, validation_batch_size)

    with tracking.start_run(run_name=run_name, enabled=mlflow_enabled) as active_run:
        run_id = getattr(getattr(active_run, "info", None), "run_id", None)

        tracking.set_tags({
            "project": "Human Protein Atlas Image Classification",
            "task": "multi-label classification",
            "model": "CustomCNN",
            "run_name": run_name,
        }, enabled=mlflow_enabled)

        tracking.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
            "image_size": IMG_SIZE,
            "input_channels": INPUT_CHANNELS,
            "num_classes": NUM_CLASSES,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "random_seed": RANDOM_STATE,
            "validation_batch_size": validation_batch_size,
            "validation_size": VALIDATION_SIZE,
            "device": str(DEVICE),
            "architecture": "CustomCNN",
            "img_size": IMG_SIZE,
            "random_state": RANDOM_STATE,
            "loss": "BCEWithLogitsLoss",
        }, enabled=mlflow_enabled)
        tracking.log_params({
            f"env_{key}": str(value) for key, value in environment.items()
        }, enabled=mlflow_enabled)
        tracking.log_json(model_config, "config/model_config.json", enabled=mlflow_enabled)
        tracking.log_json(environment, "config/environment.json", enabled=mlflow_enabled)

        best_macro_f1 = float("-inf")
        best_epoch = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            model.eval()
            val_running_loss = 0.0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > PREDICTION_THRESHOLD).astype(int)

                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            validation_loss = val_running_loss / len(valid_loader)
            all_preds = np.asarray(all_preds)
            all_labels = np.asarray(all_labels)

            macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Macro F1 Score: {macro_f1:.4f}")
            print(f"Micro F1 Score: {micro_f1:.4f}")
            print("-" * 50)

            tracking.log_metrics({
                "train_loss": train_loss,
                "training_loss": train_loss,
                "val_loss": validation_loss,
                "validation_loss": validation_loss,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
            }, step=epoch + 1, enabled=mlflow_enabled)

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_epoch = epoch + 1
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print("Best model saved!")

        if checkpoint_path.exists():
            tracking.log_artifact(
                checkpoint_path,
                artifact_path=MLFLOW_CHECKPOINT_ARTIFACT_DIR,
                enabled=mlflow_enabled,
            )
            tracking.log_model(
                load_classification_model(checkpoint_path, torch.device("cpu")),
                enabled=mlflow_enabled,
                input_example=torch.zeros(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE),
            )

        tracking.log_metrics({
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
        }, enabled=mlflow_enabled)

    print("Training completed successfully")
    print(f"Best Macro F1 Score: {best_macro_f1:.4f}")
    return {
        "run_id": run_id,
        "run_name": run_name,
        "best_macro_f1": float(best_macro_f1),
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    train_loader, valid_loader = build_data_loaders()
    result = run_training(train_loader, valid_loader)
    if result["run_id"]:
        print(f"MLflow Run ID: {result['run_id']}")


if __name__ == "__main__":
    main()
