import torch
from torch import nn
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax

from .base import EquiVectors
from ..backbone.mlp import MLP
from ..utils.lorentz import lorentz_squarednorm
from ..utils.utils import (
    build_edge_index_fully_connected,
    get_edge_index_from_ptr,
    get_edge_attr,
    get_ptr_from_batch,
)


class EquiEdgeConv(MessagePassing):
    def __init__(
        self,
        in_vectors,
        out_vectors,
        num_scalars,
        hidden_channels,
        num_layers_mlp,
        include_edges=True,
        operation="add",
        nonlinearity="exp",
        fm_norm=False,
        dropout_prob=None,
        aggr="sum",
        layer_norm=False,
    ):
        """Equivariant edge convolution, implemented using PyTorch Geometric's MessagePassing class.

        Parameters
        ----------
        in_vectors : int
            Number of input vectors.
        out_vectors : int
            Number of output vectors.
        num_scalars : int
            Number of scalar features per particle.
        hidden_channels : int
            Number of hidden channels in the MLP.
        num_layers_mlp : int
            Number of hidden layers in the MLP.
        include_edges : bool, optional
            Whether to include edge attributes in the message passing. If True, edge attributes will be calculated from fourmomenta and standardized. Default is True.
        operation : str, optional
            Operation to perform on the fourmomenta. Options are "add", "diff", or "single". Default is "add".
        nonlinearity : str, optional
            Nonlinearity to apply to the output of the MLP. Options are None, "exp", "softplus" and "softmax".
        """
        super().__init__(aggr=aggr, flow="target_to_source")
        assert num_scalars > 0 or include_edges
        self.include_edges = include_edges
        self.layer_norm = layer_norm
        self.operation = get_operation(operation)
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.fm_norm = fm_norm
        assert not (operation == "single" and fm_norm)  # unstable

        in_edges = in_vectors if include_edges else 0
        in_channels = 2 * num_scalars + in_edges
        self.mlp = MLP(
            in_shape=[in_channels],
            out_shape=out_vectors,
            hidden_channels=hidden_channels,
            hidden_layers=num_layers_mlp,
            dropout_prob=dropout_prob,
        )

        if include_edges:
            self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
            self.register_buffer("edge_mean", torch.tensor(0.0))
            self.register_buffer("edge_std", torch.tensor(1.0))

    def forward(self, fourmomenta, scalars, edge_index, batch=None):
        """
        Parameters
        ----------
        fourmomenta : torch.Tensor
            Tensor of shape (num_particles, in_vectors*4) containing the fourmomenta of the particles.
        scalars : torch.Tensor
            Tensor of shape (num_particles, num_scalars) containing scalar features for each particle.
        edge_index : torch.Tensor
            Edge index tensor containing the indices of the source and target nodes, shape (2, num_edges).
        batch : torch.Tensor, optional
            Batch tensor indicating the batch each particle belongs to. If None, all particles are assumed to belong to the same batch.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_particles, out_vectors*4) containing the predicted vectors for each edge.
        """
        # calculate and standardize edge attributes
        fourmomenta = fourmomenta.reshape(scalars.shape[0], -1, 4)
        if self.include_edges:
            edge_attr = get_edge_attr(fourmomenta, edge_index)
            if not self.edge_inited:
                self.edge_mean = edge_attr.mean().detach()
                self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
                self.edge_inited.fill_(True)
            edge_attr = (edge_attr - self.edge_mean) / self.edge_std
            edge_attr = edge_attr.reshape(edge_attr.shape[0], -1)

            # related to fourmomenta_float64 option
            edge_attr = edge_attr.to(scalars.dtype)
        else:
            edge_attr = None

        fourmomenta = fourmomenta[:, 0, :]
        # message-passing
        vecs = self.propagate(
            edge_index, s=scalars, fm=fourmomenta, edge_attr=edge_attr, batch=batch
        )
        # equivariant layer normalization
        if self.layer_norm:
            norm = lorentz_squarednorm(vecs.reshape(fourmomenta.shape[0], -1, 4))
            norm = norm.sum(dim=-1).unsqueeze(-1)
            vecs = vecs / norm.clamp(min=1e-5).sqrt()
        return vecs

    def message(self, edge_index, s_i, s_j, fm_i, fm_j, edge_attr=None):
        """
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor containing the indices of the source and target nodes, shape (2, num_edges).
        s_i : torch.Tensor
            Scalar features of the source nodes, shape (num_edges, num_scalars).
        s_j : torch.Tensor
            Scalar features of the target nodes, shape (num_edges, num_scalars).
        fm_i : torch.Tensor
            Fourmomentum of the source nodes, shape (num_edges, in_vectors*4).
        fm_j : torch.Tensor
            Fourmomentum of the target nodes, shape (num_edges, in_vectors*4).
        edge_attr : torch.Tensor, optional
            Edge attributes tensor. If None, no edge attributes will be used, shape (num_edges, num_edge_attributes).

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_edges, out_vectors*4) containing the predicted vectors for each edge.
        """
        fm_rel = self.operation(fm_i, fm_j)
        # should not be used with operation "single"
        if self.fm_norm:
            fm_rel_norm = lorentz_squarednorm(fm_rel).unsqueeze(-1)
            fm_rel_norm = fm_rel_norm.abs().sqrt().clamp(min=1e-6)
        else:
            fm_rel_norm = 1.0

        prefactor = torch.cat([s_i, s_j], dim=-1)
        if edge_attr is not None:
            prefactor = torch.cat([prefactor, edge_attr], dim=-1)
        prefactor = self.mlp(prefactor)
        prefactor = self.nonlinearity(prefactor, batch=edge_index[0])
        fm_rel = (fm_rel / fm_rel_norm)[:, None, :4]
        prefactor = prefactor.unsqueeze(-1)
        out = prefactor * fm_rel
        out = out.reshape(out.shape[0], -1)
        return out


class EquiMLP(EquiVectors):
    def __init__(
        self,
        n_vectors,
        num_blocks,
        *args,
        hidden_vectors=1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_vectors : int
            Number of output vectors per particle.
        num_blocks : int
            Number of EquiEdgeConv blocks to use in the network.
        hidden_vectors : int, optional
            Number of hidden vectors in each EquiEdgeConv block. Default is 1.
        *args
        **kwargs
        """
        super().__init__()

        assert num_blocks >= 1
        in_vectors = [1] + [hidden_vectors] * (num_blocks - 1)
        out_vectors = [hidden_vectors] * (num_blocks - 1) + [n_vectors]
        self.blocks = nn.ModuleList(
            [
                EquiEdgeConv(
                    in_vectors=in_vectors[i],
                    out_vectors=out_vectors[i],
                    *args,
                    **kwargs,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, fourmomenta, scalars=None, ptr=None):
        """
        Parameters
        ----------
        fourmomenta : torch.Tensor
            Tensor of shape (..., 4) containing the fourmomenta of the particles.
        scalars : torch.Tensor, optional
            Tensor of shape (..., num_scalars) containing scalar features for each particle. If None, a tensor of zeros will be created.
        ptr : torch.Tensor, optional
            Pointer tensor indicating the start and end of each batch for sparse tensors.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., n_vectors, 4) containing the predicted vectors for each particle.
        """
        # get edge_index and batch from ptr
        in_shape = fourmomenta.shape[:-1]
        if scalars is None:
            scalars = torch.zeros_like(fourmomenta[..., []])
        if len(in_shape) > 1:
            assert ptr is None, "ptr only supported for sparse tensors"
            edge_index, batch = build_edge_index_fully_connected(fourmomenta)
            fourmomenta = fourmomenta.reshape(math.prod(in_shape), 4)
            scalars = scalars.reshape(math.prod(in_shape), scalars.shape[-1])
        else:
            if ptr is None:
                # assume batch contains only one particle
                ptr = torch.tensor([0, len(fourmomenta)], device=fourmomenta.device)
            edge_index = get_edge_index_from_ptr(ptr)
            batch = None

        # pass through blocks
        for block in self.blocks:
            fourmomenta = block(
                fourmomenta, scalars=scalars, edge_index=edge_index, batch=batch
            )
        fourmomenta = fourmomenta.reshape(*in_shape, -1, 4)
        return fourmomenta

def safe_softmax(x, ptr):
    """Custom softmax implementation to control numerics."""
    seg_id = torch.arange(ptr.numel() - 1, device=x.device).repeat_interleave(
        ptr[1:] - ptr[:-1]
    )

    # rescale argument to avoid exp(large number)
    seg_max = scatter(x, seg_id, reduce="max")[seg_id].detach()
    z = x - seg_max

    # clamp to avoid rounding small values to zero (causes 'DivBackward0 returns nan')
    # this step is not included in standard softmax implementations
    z = z.clamp(min=-10)  # -10 works well; -20 and -5 already worse

    # actual softmax
    num = z.exp()
    den = scatter(num, seg_id, reduce="sum")[seg_id]
    out = num / den
    return out

def get_operation(operation):
    """
    Parameters
    ----------
    operation : str
        Operation to perform on the fourmomenta. Options are "add", "diff", or "single".

    Returns
    -------
    callable
        A function that performs the specified operation on two fourmomenta tensors.
    """
    if operation == "diff":
        return torch.sub
    elif operation == "add":
        return torch.add
    elif operation == "single":
        return lambda fm_i, fm_j: fm_j
    else:
        raise ValueError(
            f"Invalid operation {operation}. Options are (add, diff, single)."
        )

def get_nonlinearity(nonlinearity):
    """
    Parameters
    ----------
    nonlinearity : str or None
        Nonlinearity to apply to the output of the MLP. Options are None, "exp", "softplus", "softmax".

    Returns
    -------
    callable
        A function that applies the specified nonlinearity to the input tensor.
    """
    if nonlinearity == None:
        return lambda x, batch: x
    elif nonlinearity == "exp":
        return lambda x, batch: torch.clamp(x, min=-10, max=10).exp()
    elif nonlinearity == "softplus":
        return lambda x, batch: torch.nn.functional.softplus(x)
    elif nonlinearity == "softmax":

        def func(x, batch):
            ptr = get_ptr_from_batch(batch)
            return softmax(x, ptr=ptr)

        return func
    elif nonlinearity == "softmax_safe":

        def func(x, batch):
            ptr = get_ptr_from_batch(batch)
            return safe_softmax(x, ptr=ptr)

        return func
    else:
        raise ValueError(
            f"Invalid nonlinearity {nonlinearity}. Options are (None, exp, softplus, softmax)."
        )