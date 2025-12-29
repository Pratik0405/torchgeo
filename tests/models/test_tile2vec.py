import torch

from torchgeo.models import tile2vec_resnet18


def test_tile2vec_resnet18() -> None:
    model = tile2vec_resnet18()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 512)
