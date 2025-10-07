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
    def __init__(
        self,
        num_blocks,
        hidden_channels,
        increase_hidden_channels,
        in_rank0=0,
        in_rank1=0,
        in_rank2=1,
        out_rank=0,
        out_channels=1,
        factorize=True,
        activation="gelu",
        aggr="mean",
        compile=False,
        checkpoint_blocks=False,
    ):
        super().__init__()
        layer_kwargs = dict(factorize=factorize, aggr=aggr, compile=compile)

        # embed inputs into edge features
        self.in_aggregator_rank1 = (
            Aggregator1to2(
                in_channels=in_rank1,
                out_channels=in_rank1,
                **layer_kwargs,
            )
            if in_rank1 > 0
            else None
        )
        self.in_aggregator_rank0 = (
            Aggregator0to2(
                in_channels=in_rank0,
                out_channels=in_rank0,
                **layer_kwargs,
            )
            if in_rank0 > 0
            else None
        )

        in_rank2_effective = in_rank2 + in_rank1 + in_rank0
        assert in_rank2_effective > 0
        self.in_aggregator_rank2 = Aggregator2to2(
            in_channels=in_rank2_effective,
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
                    **layer_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )

        # extract outputs from edge features
        if out_rank == 0:
            self.out_aggregator = Aggregator2to0(
                in_channels=hidden_channels,
                out_channels=out_channels,
                **layer_kwargs,
            )
        elif out_rank == 1:
            self.out_aggregator = Aggregator2to1(
                in_channels=hidden_channels,
                out_channels=out_channels,
                **layer_kwargs,
            )
        elif out_rank == 2:
            self.out_aggregator = Aggregator2to2(
                in_channels=hidden_channels,
                out_channels=out_channels,
                **layer_kwargs,
            )
        else:
            raise NotImplementedError

    def forward(
        self, in_rank2, edge_index, batch, in_rank1=None, in_rank0=None, num_graphs=None
    ):
        # check that diagonal edges are present (required for 2to2 aggregator)
        N = batch.size(0)
        row, col = edge_index
        is_diag = row == col
        assert is_diag.sum() == N, "PELICAN requires self-loops in the graph"

        # embed inputs into edge features
        edges = [in_rank2]
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
            if self._checkpoint_blocks:
                x = checkpoint(
                    block,
                    x,
                    edge_index=edge_index,
                    batch=batch,
                    use_reentrant=False,
                )
            else:
                x = block(x, edge_index=edge_index, batch=batch)

        # extract outputs from edge features
        out = self.out_aggregator(x, edge_index=edge_index, batch=batch, G=num_graphs)
        return out
