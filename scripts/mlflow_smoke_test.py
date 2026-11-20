import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Run a synthetic MLflow smoke experiment.")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default=None)
    return parser.parse_args()


def load_training_module():
    path = ROOT_DIR / "human-protein-classification.py"
    spec = importlib.util.spec_from_file_location("hpa_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_synthetic_loaders(samples, batch_size, seed):
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader

    from config import IMG_SIZE, NUM_CLASSES
    from preprocessing import create_multilabel, stack_channels

    class SyntheticProteinDataset(Dataset):
        def __init__(self, count, sample_seed):
            rng = np.random.default_rng(sample_seed)
            self.samples = []
            for index in range(count):
                channels = [
                    rng.integers(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                    for _ in range(4)
                ]
                image = stack_channels(channels)
                label = create_multilabel(
                    f"{index % NUM_CLASSES} {(index * 7 + 3) % NUM_CLASSES}",
                    NUM_CLASSES,
                )
                self.samples.append((image, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            image, label = self.samples[idx]
            return torch.from_numpy(image), torch.as_tensor(label, dtype=torch.float32)

    train_dataset = SyntheticProteinDataset(samples, seed)
    valid_dataset = SyntheticProteinDataset(max(2, samples // 4), seed + 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def main():
    args = parse_args()
    os.environ["MLFLOW_ENABLED"] = "1"
    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    if args.experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment

    training = load_training_module()
    train_loader, valid_loader = build_synthetic_loaders(
        args.samples, args.batch_size, seed=20261120
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    with tempfile.TemporaryDirectory(prefix="hpa-mlflow-smoke-") as temp_dir:
        for index in range(args.runs):
            result = training.run_training(
                train_loader,
                valid_loader,
                epochs=args.epochs,
                learning_rate=0.001,
                batch_size=args.batch_size,
                validation_batch_size=args.batch_size,
                checkpoint_path=Path(temp_dir) / f"smoke_run_{index + 1}.pth",
                run_name=f"smoke-{timestamp}-{index + 1}",
                mlflow_enabled=True,
            )
            print("SMOKE_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
