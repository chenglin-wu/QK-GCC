# Copyright (c) 2026 jinkz33@XMU&UofG

import torch

from qkgcc import QKGCC


def test_output_shape() -> None:
    model = QKGCC().eval()
    x1, x2 = torch.randn(2, 2048), torch.randn(2, 2048)
    with torch.inference_mode():
        logits = model(x1, x2)
    assert logits.shape == (2, 47)


def test_prediction_range() -> None:
    model = QKGCC().eval()
    with torch.inference_mode():
        delay = model.predict(torch.randn(1, 2048), torch.randn(1, 2048))
    assert -23 <= delay.item() <= 23

