"""Baseline LLoCa-GNN."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .lloca_message_passing import LLoCaMessagePassing
from .mlp import MLP
from ..reps.tensorreps import TensorReps


class EdgeConv(LLoCaMessagePassing):
    def __init__(
        self,
        reps,
        num_layers_mlp1,
        num_layers_mlp2,
        aggr="add",
        num_edge_attr=0,
        dropout_prob=None,
    ):
        """Simple edge convolution layer.

        Parameters
        ----------
        reps : TensorReps
            Tensor representation used during message passing.
        num_layers_mlp1 : int
            Number of hidden layers in the first MLP.
        num_layers_mlp2 : int
            Number of hidden layers in the second MLP.
            If 0, no second MLP is used.
        aggr : str
            Aggregation method. One of "add", "mean", or "max".
        num_edge_attr : int
            Number of edge attributes.
        dropout_prob : float
            Dropout probability in the MLPs.
        """
        super().__init__(aggr=aggr, params_dict={"x": {"type": "local", "rep": reps}})
        self.mlp1 = MLP(
            in_shape=[reps.dim * 2 + num_edge_attr],
            out_shape=[reps.dim],
            hidden_layers=num_layers_mlp1,
            hidden_channels=reps.dim,
            dropout_prob=dropout_prob,
        )
        self.mlp2 = (
            MLP(
                in_shape=[reps.dim],
                out_shape=[reps.dim],
                hidden_layers=num_layers_mlp2,
                hidden_channels=reps.dim,
                dropout_prob=dropout_prob,
            )
            if num_layers_mlp2 > 0
            else nn.Identity()
        )

    def forward(self, x, frames, edge_index, batch=None, edge_attr=None):
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input data with shape (num_items, reps.dim)
        frames : Frames
            Local frames used for message passing
        edge_index : Tensor
            Edge index tensor with shape (2, num_edges)
        batch : Tensor
            Batch tensorwith shape (num_items,)

        Returns
        -------
        x_aggr : Tensor
            Outputs with shape (num_items, reps.dim)
        """
        frames = (frames, frames)

        x_aggr = self.propagate(
            edge_index,
            x=x,
            frames=frames,
            edge_attr=edge_attr,
            batch=batch,
        )
        x_aggr = self.mlp2(x_aggr)
        return x_aggr

    def message(self, x_i, x_j, frames_i, frames_j, edge_attr=None):
        x = x_j
        x = torch.cat((x, x_i), dim=-1)
        if edge_attr is not None:
            x = torch.cat((x, edge_attr), dim=-1)
        x = self.mlp1(x)
        return x


class GraphNet(nn.Module):
    """Baseline LLoCa-GNN.

    Simple message-passing graph neural network, consisting of EdgeConv blocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_reps : str
        Tensor representation used in the hidden layers.
    out_channels : int
        Number of output channels.
    num_blocks : int
        Number of EdgeConv blocks.
    *args
    checkpoint_blocks : bool
        Whether to use gradient checkpointing in the EdgeConv blocks.
    **kwargs
    """

    def __init__(
        self,
        in_channels: int,
        hidden_reps: str,
        out_channels: int,
        num_blocks: int,
        *args,
        checkpoint_blocks=False,
        **kwargs,
    ):
        super().__init__()
        hidden_reps = TensorReps(hidden_reps)
        self.checkpoint_blocks = checkpoint_blocks

        self.linear_in = nn.Linear(in_channels, hidden_reps.dim)
        self.linear_out = nn.Linear(hidden_reps.dim, out_channels)
        self.blocks = nn.ModuleList(
            [
                EdgeConv(
                    hidden_reps,
                    *args,
                    **kwargs,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, inputs, frames, edge_index, batch=None, edge_attr=None):
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data with shape (num_items, in_channels)
        frames : Frames
            Local frames used for message passing
        edge_index : Tensor
            Edge index tensor with shape (2, num_edges)
        batch : Tensor
            Batch tensorwith shape (num_items,)
            If None, assumes fully connected graph along the num_items direction.
        edge_attr : Tensor
            Edge attribute tensor with shape (num_edges, num_edge_attr)

        Returns
        -------
        outputs : Tensor
            Outputs with shape (num_items, out_channels)
        """
        x = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                x = checkpoint(
                    block,
                    x=x,
                    frames=frames,
                    edge_index=edge_index,
                    batch=batch,
                    edge_attr=edge_attr,
                )
            else:
                x = block(
                    x=x,
                    frames=frames,
                    edge_index=edge_index,
                    batch=batch,
                    edge_attr=edge_attr,
                )
        outputs = self.linear_out(x)
        return outputs
