"""Base class for equivariant vector predictors"""
from torch import nn


class EquiVectors(nn.Module):
    """Abstract class for equivariantly predicting vectors
    based on fourmomenta and scalars"""

    def __init__(self):
        super().__init__()

    def forward(self, fourmomenta, scalars, *args, **kwargs):
        raise NotImplementedError
