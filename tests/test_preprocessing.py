import cv2
import numpy as np
import pytest

import config
from preprocessing import (
    create_multilabel,
    decode_channel_bytes,
    load_channel_image,
    stack_channels,
)


def _png_bytes(array):
    ok, buffer = cv2.imencode(".png", array)
    assert ok
    return buffer.tobytes()


def test_decode_channel_bytes_resizes_to_grayscale():
    source = np.full((64, 96), 200, dtype=np.uint8)
    decoded = decode_channel_bytes(_png_bytes(source))

    assert decoded.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert decoded.dtype == np.uint8


def test_decode_channel_bytes_rejects_invalid_bytes():
    with pytest.raises(ValueError, match="Could not decode"):
        decode_channel_bytes(b"not an image")


def test_load_channel_image(tmp_path):
    path = tmp_path / "sample_red.png"
    cv2.imwrite(str(path), np.zeros((32, 32), dtype=np.uint8))

    loaded = load_channel_image(path)
    assert loaded.shape == (config.IMG_SIZE, config.IMG_SIZE)


def test_stack_channels_normalises_and_validates():
    channels = [np.full((config.IMG_SIZE, config.IMG_SIZE), 255, dtype=np.uint8)] * 4
    stacked = stack_channels(channels)

    assert stacked.shape == (4, config.IMG_SIZE, config.IMG_SIZE)
    assert stacked.dtype == np.float32
    assert float(stacked.min()) == 1.0
    assert float(stacked.max()) == 1.0

    with pytest.raises(ValueError, match="Exactly four channels"):
        stack_channels(channels[:3])


def test_create_multilabel():
    label = create_multilabel("0 27")
    assert label.shape == (28,)
    assert label[0] == 1.0
    assert label[27] == 1.0
    assert label.sum() == 2.0
