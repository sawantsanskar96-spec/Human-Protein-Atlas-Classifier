import torch

import config
from cnn_model import CustomCNN, load_model


def test_state_dict_matches_checkpoint():
    checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    model = CustomCNN()

    assert list(model.state_dict().keys()) == list(checkpoint.keys())
    for key in model.state_dict():
        assert tuple(model.state_dict()[key].shape) == tuple(checkpoint[key].shape)


def test_load_model_is_evaluated_and_returns_28_logits():
    model = load_model(device=torch.device("cpu"))

    assert isinstance(model, CustomCNN)
    assert model.training is False

    output = model(torch.zeros(1, 4, config.IMG_SIZE, config.IMG_SIZE))
    assert tuple(output.shape) == (1, config.NUM_CLASSES)
