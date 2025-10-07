import math
import torch
from torch import nn

from .primitives import (
    bell_number,
    aggregate_0to2,
    aggregate_1to2,
    aggregate_2to0,
    aggregate_2to1,
    aggregate_2to2,
)

ACTIVATION = {
    "leaky_relu": nn.LeakyReLU(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
}


class GeneralAggregator(nn.Module):
    def __init__(
        self,
        in_rank,
        out_rank,
        in_channels,
        out_channels,
        factorize=True,
        aggr="mean",
    ):
        super().__init__()
        num_maps = bell_number(in_rank + out_rank)
        self.aggr = aggr
        self.aggregator = None
        self.factorize = factorize

        if factorize:
            self.coeffs00 = nn.Parameter(torch.empty(in_channels, num_maps))
            self.coeffs01 = nn.Parameter(torch.empty(out_channels, num_maps))
            self.coeffs10 = nn.Parameter(torch.empty(in_channels, out_channels))
            self.coeffs11 = nn.Parameter(torch.empty(in_channels, out_channels))
            scale0 = math.sqrt(3.0 / num_maps)
            scale1 = math.sqrt(3.0 / in_channels)
            torch.nn.init.uniform_(self.coeffs00, a=-scale0, b=scale0)
            torch.nn.init.uniform_(self.coeffs01, a=-scale0, b=scale0)
            torch.nn.init.uniform_(self.coeffs10, a=-scale1, b=scale1)
            torch.nn.init.uniform_(self.coeffs11, a=-scale1, b=scale1)
        else:
            self.coeffs_direct = nn.Parameter(
                torch.empty(in_channels, out_channels, num_maps)
            )
            scale = math.sqrt(6.0 / (in_channels * num_maps))
            torch.nn.init.uniform_(self.coeffs_direct, a=-scale, b=scale)

    @property
    def coeffs(self):
        if self.factorize:
            return self.coeffs00.unsqueeze(1) * self.coeffs10.unsqueeze(
                2
            ) + self.coeffs01.unsqueeze(0) * self.coeffs11.unsqueeze(2)
        else:
            return self.coeffs_direct

    def forward(self, x, *args, **kwargs):
        x = self.aggregator(x, *args, reduce=self.aggr, **kwargs)
        M, C, N = x.shape
        x_flat = x.reshape(M, C * N)
        coeffs_flat = self.coeffs.permute(0, 1, 2).reshape(-1, C * N).contiguous()
        out = torch.nn.functional.linear(x_flat, coeffs_flat)
        return out


class Aggregator2to2(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(2, 2, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_2to2


class Aggregator2to1(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(2, 1, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_2to1


class Aggregator2to0(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(2, 0, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_2to0


class Aggregator1to2(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(1, 2, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_1to2


class Aggregator0to2(GeneralAggregator):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(0, 2, in_channels, out_channels, **kwargs)
        self.aggregator = aggregate_0to2


class PELICANBlock(nn.Module):
    def __init__(
        self,
        hidden_channels,
        increase_hidden_channels=1.0,
        activation="gelu",
        **kwargs,
    ):
        super().__init__()
        hidden_channels_2 = int(increase_hidden_channels * hidden_channels)
        linear_in = nn.Linear(hidden_channels, hidden_channels_2)
        norm = nn.LayerNorm(normalized_shape=hidden_channels_2)
        self.activation = ACTIVATION[activation]
        self.mlp = nn.ModuleList([linear_in, self.activation, norm])

        self.aggregator = Aggregator2to2(
            in_channels=hidden_channels_2,
            out_channels=hidden_channels,
            **kwargs,
        )

    def forward(self, x, edge_index, batch, **kwargs):
        for layer in self.mlp:
            x = layer(x)

        x = self.aggregator(x, edge_index=edge_index, batch=batch, **kwargs)
        x = self.activation(x)
        return x
