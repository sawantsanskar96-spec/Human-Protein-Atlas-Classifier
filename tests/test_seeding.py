import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_training_module():
    path = ROOT_DIR / "human-protein-classification.py"
    spec = importlib.util.spec_from_file_location("hpa_training_seed_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_set_seed_reproduces_random_streams():
    training = _load_training_module()

    training.set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1))

    training.set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.allclose(first[2], second[2])
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
