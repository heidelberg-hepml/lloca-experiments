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
    """General aggregator class for aggregating features between different ranks."""

    def __init__(
        self,
        aggregator,
        in_rank,
        out_rank,
        in_channels,
        out_channels,
        map_multipliers=True,
        factorize=False,
        aggr="mean",
    ):
        """
        Parameters
        ----------
        aggregator : callable
            Aggregation function to use.
        in_rank : int
            Input rank (0 for graph, 1 for nodes, 2 for edges).
        out_rank : int
            Output rank (0 for graph, 1 for nodes, 2 for edges).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        map_multipliers : bool
            Whether to use learnable multipliers for each aggregation map, by default True.
        factorize : bool
            Whether to use factorized coefficients, by default False.
            Factorization reduces the number of parameters.
        aggr : str
            Aggregation method to use ('mean', 'sum', 'prod', 'amin', 'amax'), by default 'mean'.
        """
        super().__init__()
        self.num_maps = bell_number(in_rank + out_rank)
        self.in_channels = in_channels

        self.aggr = aggr
        self.aggregator = aggregator
        self.factorize = factorize

        self.map_multipliers = (
            nn.Parameter(torch.ones(self.num_maps)) if map_multipliers else None
        )

        if factorize:
            self.coeffs00 = nn.Parameter(torch.empty(in_channels, self.num_maps))
            self.coeffs01 = nn.Parameter(torch.empty(out_channels, self.num_maps))
            self.coeffs10 = nn.Parameter(torch.empty(in_channels, out_channels))
            self.coeffs11 = nn.Parameter(torch.empty(in_channels, out_channels))
            nn.init.kaiming_uniform_(self.coeffs00, nonlinearity="linear")
            nn.init.kaiming_uniform_(self.coeffs01, nonlinearity="linear")
            nn.init.kaiming_uniform_(self.coeffs10, nonlinearity="linear")
            nn.init.kaiming_uniform_(self.coeffs11, nonlinearity="linear")
        else:
            self.coeffs_direct = nn.Parameter(
                torch.empty(in_channels, out_channels, self.num_maps)
            )
            nn.init.kaiming_uniform_(self.coeffs_direct, nonlinearity="relu")

    @property
    def coeffs(self):
        if self.factorize:
            coeffs = self.coeffs00.unsqueeze(1) * self.coeffs10.unsqueeze(
                2
            ) + self.coeffs01.unsqueeze(0) * self.coeffs11.unsqueeze(2)
        else:
            coeffs = self.coeffs_direct
        if self.map_multipliers is not None:
            coeffs = coeffs * self.map_multipliers.view(1, 1, self.num_maps)
        return coeffs

    def forward(self, x, *args, **kwargs):
        """Forward pass of the aggregator.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (in_objects, in_channels).
        *args
            Additional arguments to pass to the aggregator.
        **kwargs

        Returns
        -------
        out : torch.Tensor
            Output features of shape (out_objects, in_channels).
        """
        x = self.aggregator(x, *args, reduce=self.aggr, **kwargs)

        in_objects, in_channels, num_maps = x.shape
        x_flat = x.reshape(in_objects, in_channels * num_maps)
        coeffs_flat = self.coeffs.reshape(-1, in_channels * num_maps).contiguous()
        out = torch.nn.functional.linear(x_flat, coeffs_flat)
        return out


class Aggregator2to2(GeneralAggregator):
    """Aggregator from edges (rank 2) to edges (rank 2)."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(aggregate_2to2, 2, 2, in_channels, out_channels, **kwargs)


class Aggregator2to1(GeneralAggregator):
    """Aggregator from edges (rank 2) to nodes (rank 1)."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(aggregate_2to1, 2, 1, in_channels, out_channels, **kwargs)


class Aggregator2to0(GeneralAggregator):
    """Aggregator from edges (rank 2) to graph (rank 0)."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(aggregate_2to0, 2, 0, in_channels, out_channels, **kwargs)


class Aggregator1to2(GeneralAggregator):
    """Aggregator from nodes (rank 1) to edges (rank 2)."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(aggregate_1to2, 1, 2, in_channels, out_channels, **kwargs)


class Aggregator0to2(GeneralAggregator):
    """Aggregator from graph (rank 0) to edges (rank 2)."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(aggregate_0to2, 0, 2, in_channels, out_channels, **kwargs)


class PELICANBlock(nn.Module):
    """A single PELICAN block consisting of a feedforward network and edge-to-edge aggregation."""

    def __init__(
        self,
        hidden_channels,
        increase_hidden_channels=1.0,
        activation="gelu",
        dropout_prob=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        hidden_channels : int
            Number of hidden channels.
        increase_hidden_channels : float
            Factor to increase hidden channels in the feedforward network, by default 1.0.
        activation : str
            Activation function to use ('gelu', 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'silu'), by default 'gelu'.
        dropout_prob: float
            Dropout probability, by default None.
        **kwargs
        """
        super().__init__()
        hidden_channels_2 = int(increase_hidden_channels * hidden_channels)
        linear_in = nn.Linear(hidden_channels, hidden_channels_2)
        self.activation = ACTIVATION[activation]
        norm = nn.RMSNorm(normalized_shape=hidden_channels_2)
        dropout = (
            nn.Dropout(p=dropout_prob) if dropout_prob is not None else nn.Identity()
        )
        self.mlp = nn.ModuleList([linear_in, self.activation, norm, dropout])

        self.aggregator = Aggregator2to2(
            in_channels=hidden_channels_2,
            out_channels=hidden_channels,
            **kwargs,
        )

    def forward(self, x, edge_index, batch, **kwargs):
        """Forward pass of the PELICAN block.

        Parameters
        ----------
        x : torch.Tensor
            Edge-level features of shape (E, C).
        edge_index : torch.Tensor
            Edge index tensor of shape (2, E).
        batch : torch.Tensor
            Batch tensor of shape (N).
        **kwargs

        Returns
        -------
        x : torch.Tensor
            Updated edge-level features of shape (E, C).
        """
        for layer in self.mlp:
            x = layer(x)

        x = self.aggregator(x, edge_index=edge_index, batch=batch, **kwargs)
        x = self.activation(x)
        return x
