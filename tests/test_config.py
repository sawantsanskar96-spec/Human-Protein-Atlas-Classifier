import os
import subprocess
import sys
from pathlib import Path

import config


def test_class_labels_are_complete_and_unique():
    assert len(config.CLASS_LABELS) == 28
    assert len(set(config.CLASS_LABELS)) == 28
    assert config.CLASS_LABELS[0] == "Nucleoplasm"
    assert config.CLASS_LABELS[-1] == "Rods & rings"


def test_defaults_preserve_training_behaviour():
    assert config.IMG_SIZE == 128
    assert config.NUM_CLASSES == 28
    assert config.PREDICTION_THRESHOLD == 0.3
    assert config.BATCH_SIZE == 16
    assert config.VALIDATION_SIZE == 0.2
    assert config.RANDOM_STATE == 42


def test_checkpoint_default_points_to_existing_model():
    assert config.MODEL_PATH.name == "protein_model.pth"
    assert config.MODEL_PATH.exists()


def test_model_path_environment_override(tmp_path):
    expected = tmp_path / "custom_model.pth"
    expected.write_bytes(b"placeholder")
    env = os.environ.copy()
    env["PROTEIN_MODEL_PATH"] = str(expected)

    result = subprocess.run(
        [sys.executable, "-c", "import config; print(config.MODEL_PATH)"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert Path(result.stdout.strip()) == expected
