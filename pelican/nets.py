import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .layers import (
    Aggregator0to2,
    Aggregator1to2,
    Aggregator2to2,
    Aggregator2to1,
    Aggregator2to0,
    PELICANBlock,
)


class PELICAN(nn.Module):
    """PELICAN network.

    PELICAN stands for Permutation Equivariant and Lorentz Invariant or Covariant Aggregator Network.
    PELICAN takes input of varying rank (0, 1, or 2), projects them onto rank 2 objects (edges),
    processes them with multiple PELICANBlock, and extracts output of varying rank (0, 1, or 2).
    """

    def __init__(
        self,
        in_channels_rank2,
        in_channels_rank1,
        in_channels_rank0,
        out_rank,
        out_channels,
        num_blocks,
        hidden_channels,
        increase_hidden_channels=1.0,
        map_multipliers=True,
        factorize=False,
        activation="gelu",
        dropout_prob=None,
        aggr="mean",
        compile=False,
        checkpoint_blocks=False,
    ):
        """
        Parameters
        ----------
        in_channels_rank2 : int
            Edge-level input features (rank 2). Can be zero.
        in_channels_rank1 : int
            Node-level input features (rank 1). Can be zero.
        in_channels_rank0 : int
            Graph-level input features (rank 0). Can be zero.
        out_rank : int
            Rank of the output features (0, 1, or 2).
        out_channels : int
            Number of output channels.
        num_blocks : int
            Number of PELICAN blocks.
        hidden_channels : int
            Number of hidden channels.
        increase_hidden_channels : float
            Factor to increase hidden channels in the feedforward network. Default is 1.0.
        map_multipliers : bool
            Whether to use learnable multipliers for each aggregation map. Default is True.
        factorize : bool
            Whether to use factorized linear layers in the feedforward network. Default is False.
        activation : str
            Activation function to use ('gelu', 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'silu'), by default 'gelu'.
        dropout_prob: float
            Dropout probability in the feedforward network, by default None (no dropout).
        aggr : str
            Aggregation method to use ('mean', 'sum', 'prod', 'amin', 'amax'), by default 'mean'.
        compile : bool
            Whether to compile the model with torch.compile. Default is False.
            Compiling the model leads to significant speedups on GPU, because the aggregation functions involve many small operations that otherwise require many individual kernel launches.
            It is recommended to run 'model = torch.compile(model, **kwargs)' outside of the constructor, however we provide this option for convenience.
        checkpoint_blocks : bool
            Whether to use gradient checkpointing for PELICAN blocks to save memory. Default is False.
        """
        super().__init__()
        layer_kwargs = dict(
            factorize=factorize, map_multipliers=map_multipliers, aggr=aggr
        )

        # embed inputs into edge features
        self.in_aggregator_rank1 = (
            Aggregator1to2(
                in_channels=in_channels_rank1,
                out_channels=in_channels_rank1,
                **layer_kwargs,
            )
            if in_channels_rank1 > 0
            else None
        )
        self.in_aggregator_rank0 = (
            Aggregator0to2(
                in_channels=in_channels_rank0,
                out_channels=in_channels_rank0,
                **layer_kwargs,
            )
            if in_channels_rank0 > 0
            else None
        )

        in_channels = in_channels_rank2 + in_channels_rank1 + in_channels_rank0
        assert in_channels > 0
        self.in_aggregator_rank2 = Aggregator2to2(
            in_channels=in_channels,
            out_channels=hidden_channels,
            **layer_kwargs,
        )

        # process edge features
        self._checkpoint_blocks = checkpoint_blocks
        self.blocks = nn.ModuleList(
            [
                PELICANBlock(
                    hidden_channels=hidden_channels,
                    increase_hidden_channels=increase_hidden_channels,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    **layer_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )

        # extract outputs from edge features
        out_aggregator_class = {
            0: Aggregator2to0,
            1: Aggregator2to1,
            2: Aggregator2to2,
        }
        self.out_aggregator = out_aggregator_class[out_rank](
            in_channels=hidden_channels,
            out_channels=out_channels,
            **layer_kwargs,
        )

        if compile:
            # ugly hack to make torch.compile convenient for users
            # the clean solution is model = torch.compile(model, **kwargs) outside of the constructor
            self.__class__ = torch.compile(
                self.__class__, dynamic=True, fullgraph=True, mode="default"
            )

    def forward(
        self,
        edge_index,
        batch,
        in_rank2=None,
        in_rank1=None,
        in_rank0=None,
        num_graphs=None,
    ):
        """Forward pass.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor of shape (2, E).
        batch : torch.Tensor
            Batch tensor of shape (N).
        in_rank2 : torch.Tensor
            Edge-level input features of shape (E, C), by default None.
        in_rank1 : torch.Tensor
            Node-level input features of shape (N, C), by default None.
        in_rank0 : torch.Tensor
            Graph-level input features of shape (G, C), by default None.
        num_graphs : int
            Number of graphs in the batch. If None, it will be inferred from the batch tensor.
            Inferring this number from the batch tensor requires a GPU/CPU synchronization,
            which slows down the code when running on GPU.

        Returns
        -------
        out : torch.Tensor
            Output features of shape (G, C), (N, C), or (E, C) depending on out_rank.
        """
        # check that diagonal edges are present (required for 2to2 aggregator)
        N = batch.size(0)
        row, col = edge_index
        is_diag = row == col
        assert is_diag.sum() == N, "PELICAN requires self-loops in the graph"

        # embed inputs into edge features
        edges = [in_rank2] if in_rank2 is not None else []
        if in_rank1 is not None and self.in_aggregator_rank1 is not None:
            edges_fromrank1 = self.in_aggregator_rank1(in_rank1, edge_index, batch)
            edges.append(edges_fromrank1)
        if in_rank0 is not None and self.in_aggregator_rank0 is not None:
            edges_fromrank0 = self.in_aggregator_rank0(in_rank0, edge_index, batch)
            edges.append(edges_fromrank0)
        edges = torch.cat(edges, dim=-1)
        x = self.in_aggregator_rank2(edges, edge_index, batch)

        # process edge features
        for block in self.blocks:
            kwargs = dict(x=x, edge_index=edge_index, batch=batch)
            if self._checkpoint_blocks:
                x = checkpoint(block, use_reentrant=False, **kwargs)
            else:
                x = block(**kwargs)

        # extract outputs from edge features
        out = self.out_aggregator(x, edge_index=edge_index, batch=batch, G=num_graphs)
        return out
