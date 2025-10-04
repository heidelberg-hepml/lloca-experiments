import torch
import math
from torch_geometric.nn import MessagePassing

from .base import EquiVectors
from ..utils.utils import (
    get_edge_index_from_shape,
    get_edge_index_from_ptr,
    get_edge_attr,
    get_batch_from_ptr,
)
from ..utils.lorentz import lorentz_squarednorm
from .equimlp import get_operation, get_nonlinearity


class PELICANVectors(EquiVectors, MessagePassing):
    def __init__(
        self,
        n_vectors,
        num_scalars,
        net,
        operation="add",
        nonlinearity="softmax",
        aggr="sum",
        fm_norm=False,
    ):
        super().__init__(aggr=aggr)
        self.net = net(in_rank1=num_scalars, out_channels=n_vectors)

        self.register_buffer("edge_inited", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("edge_mean", torch.tensor(0.0))
        self.register_buffer("edge_std", torch.tensor(1.0))

        self.operation = get_operation(operation)
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.fm_norm = fm_norm
        assert not (operation == "single" and fm_norm)  # unstable

    def forward(self, fourmomenta, scalars=None, ptr=None):
        # move to sparse tensors
        in_shape = fourmomenta.shape[:-1]
        if scalars is None:
            scalars = torch.zeros_like(fourmomenta[..., []])
        if len(in_shape) > 1:
            assert ptr is None, "ptr only supported for sparse tensors"
            edge_index, batch = get_edge_index_from_shape(
                fourmomenta, remove_self_loops=False
            )
            fourmomenta = fourmomenta.reshape(math.prod(in_shape), 4)
            scalars = scalars.reshape(math.prod(in_shape), scalars.shape[-1])
        else:
            if ptr is None:
                # assume batch contains only one particle
                ptr = torch.tensor([0, len(fourmomenta)], device=fourmomenta.device)
            edge_index = get_edge_index_from_ptr(ptr, remove_self_loops=False)
            batch = get_batch_from_ptr(ptr)

        # compute prefactors
        edge_attr = self.get_edge_attr(fourmomenta, edge_index).to(scalars.dtype)

        # message-passing
        vecs = self.propagate(
            edge_index, fm=fourmomenta, s=scalars, edge_attr=edge_attr, batch=batch
        )
        vecs = vecs.reshape(fourmomenta.shape[0], -1, 4)

        # reshape result
        vecs = vecs.reshape(*in_shape, -1, 4)
        return vecs

    def message(self, edge_index, fm_i, fm_j, s_i, s_j, edge_attr, batch):
        # prepare fourmomenta
        fm_rel = self.operation(fm_i, fm_j)
        if self.fm_norm:
            fm_rel_norm = lorentz_squarednorm(fm_rel).unsqueeze(-1)
            fm_rel_norm = fm_rel_norm.abs().sqrt().clamp(min=1e-6)
        else:
            fm_rel_norm = 1.0
        fm_rel = (fm_rel / fm_rel_norm)[:, None, :4]

        # message-passing
        prefactor = self.net(
            in_rank2=edge_attr, in_rank1=s_i, edge_index=edge_index, batch=batch
        )
        prefactor = self.nonlinearity(prefactor, batch=edge_index[0])
        prefactor = prefactor.unsqueeze(-1)
        out = prefactor * fm_rel
        out = out.reshape(out.shape[0], -1)
        return out

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited.fill_(True)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        edge_attr = edge_attr.reshape(edge_attr.shape[0], -1)
        return edge_attr
