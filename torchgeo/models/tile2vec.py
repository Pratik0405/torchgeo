# Copyright (c) TorchGeo Contributors.
# Licensed under the MIT License.

"""Tile2Vec ResNet models."""

from typing import Any

import timm
from torch import nn
from torchvision.models._api import WeightsEnum

__all__ = ["Tile2Vec_Weights", "tile2vec_resnet18"]


class Tile2Vec_Weights(WeightsEnum):
    """Tile2Vec pretrained weights."""
    # No weights yet - placeholder for future
    pass


def tile2vec_resnet18(
    weights: Tile2Vec_Weights | None = None,
    **kwargs: Any,
) -> nn.Module:
    """ResNet-18 encoder configured for Tile2Vec."""
    if weights:
        kwargs["in_chans"] = weights.meta["in_chans"]

    model = timm.create_model(
        "resnet18",
        num_classes=0,      # remove classifier
        global_pool="avg",  # global average pooling
        **kwargs,
    )

    if weights:
        model.load_state_dict(
            weights.get_state_dict(progress=True),
            strict=False,
        )

    return model
