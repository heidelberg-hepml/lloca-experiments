import math
import torch
from torch_geometric.nn import MessagePassing

from einops import rearrange
from lgatr import embed_vector, extract_vector
from lgatr.primitives.attention import sdp_attention
from lgatr.layers import EquiLayerNorm
from lgatr.primitives.invariants import _load_inner_product_factors

from .base import EquiVectors
from .equimlp import get_operation, get_nonlinearity, get_edge_index_and_batch
from ..utils.lorentz import lorentz_squarednorm
from ..utils.utils import get_batch_from_ptr
from lloca.backbone.attention_backends.xformers_attention import (
    get_xformers_attention_mask,
)


class LGATrVectors(EquiVectors):
    def __init__(
        self,
        n_vectors,
        num_scalars,
        hidden_mv_channels,
        hidden_s_channels,
        net,
        lgatr_norm=False,
    ):
        super().__init__()
        self.n_vectors = n_vectors
        out_mv_channels = (
            2 * n_vectors * max(1, hidden_mv_channels // (2 * n_vectors))
            if hidden_mv_channels > 0
            else 0
        )
        out_s_channels = (
            2 * n_vectors * max(1, hidden_s_channels // (2 * n_vectors))
            if hidden_s_channels > 0
            else 0
        )
        self.net = net(
            in_s_channels=num_scalars,
            out_mv_channels=out_mv_channels,
            out_s_channels=out_s_channels,
        )
        self.lgatr_norm = EquiLayerNorm() if lgatr_norm else None

    def forward(self, fourmomenta, scalars=None, ptr=None, **kwargs):
        attn_kwargs = {}
        if ptr is not None:
            batch = get_batch_from_ptr(ptr)
            on_cpu = fourmomenta.device == torch.device("cpu")
            mask = get_xformers_attention_mask(
                batch, materialize=on_cpu, dtype=scalars.dtype
            )
            attn_kwargs["attn_mask" if on_cpu else "attn_bias"] = mask
            fourmomenta = fourmomenta.unsqueeze(0)
            scalars = scalars.unsqueeze(0)

        # get query and key from LGATr
        mv = embed_vector(fourmomenta).unsqueeze(-2).to(scalars.dtype)
        out_mv, out_s = self.net(mv, scalars, **attn_kwargs)
        if self.lgatr_norm is not None:
            out_mv, out_s = self.lgatr_norm(out_mv, out_s)

        # extract q and k
        q_mv, k_mv = torch.chunk(out_mv.to(fourmomenta.dtype), chunks=2, dim=-2)
        q_s, k_s = torch.chunk(out_s.to(fourmomenta.dtype), chunks=2, dim=-1)

        # unpack the n_vectors axis
        q_mv = q_mv.reshape(*q_mv.shape[:-2], self.n_vectors, -1, q_mv.shape[-1])
        k_mv = k_mv.reshape(*k_mv.shape[:-2], self.n_vectors, -1, k_mv.shape[-1])
        q_s = q_s.reshape(*q_s.shape[:-1], self.n_vectors, -1)
        k_s = k_s.reshape(*k_s.shape[:-1], self.n_vectors, -1)

        # initialize values (v_mv=fourmomenta, v_s=empty)
        v_mv = embed_vector(fourmomenta).unsqueeze(-2)
        v_mv = v_mv.unsqueeze(-3).expand(*q_mv.shape[:-2], *v_mv.shape[-2:])
        v_s = torch.empty(*q_s.shape[:-1], 0, device=q_s.device, dtype=q_s.dtype)

        # geometric attention between learned q, k and fixed v
        # transpose to make the n_vectors axis a batch axis
        q_mv, k_mv, v_mv = (
            q_mv.transpose(-3, -4),
            k_mv.transpose(-3, -4),
            v_mv.transpose(-3, -4),
        )
        q_s, k_s, v_s = (
            q_s.transpose(-2, -3),
            k_s.transpose(-2, -3),
            v_s.transpose(-2, -3),
        )
        out_mv, out_s = sdp_attention(
            q_mv=q_mv, k_mv=k_mv, q_s=q_s, k_s=k_s, v_mv=v_mv, v_s=v_s, **attn_kwargs
        )
        out_mv = out_mv.transpose(-3, -4)

        # extract vector part of multivector output
        out = extract_vector(out_mv).squeeze(-2)

        if ptr is not None:
            out = out.squeeze(0)  # undo initial unsqueeze(0)
        return out


class LGATrVectors2(EquiVectors, MessagePassing):
    def __init__(
        self,
        n_vectors,
        num_scalars,
        hidden_mv_channels,
        hidden_s_channels,
        net,
        operation="add",
        nonlinearity="softmax",
        aggr="sum",
        fm_norm=False,
        layer_norm=False,
        lgatr_norm=True,
    ):
        super().__init__(aggr=aggr)
        self.n_vectors = n_vectors
        out_mv_channels = (
            2 * n_vectors * max(1, hidden_mv_channels // (2 * n_vectors))
            if hidden_mv_channels > 0
            else 0
        )
        out_s_channels = (
            2 * n_vectors * max(1, hidden_s_channels // (2 * n_vectors))
            if hidden_s_channels > 0
            else 0
        )
        self.net = net(
            in_s_channels=num_scalars,
            out_mv_channels=out_mv_channels,
            out_s_channels=out_s_channels,
        )
        self.lgatr_norm = EquiLayerNorm() if lgatr_norm else None

        self.operation = get_operation(operation)
        self.nonlinearity = get_nonlinearity(nonlinearity)
        self.fm_norm = fm_norm
        self.layer_norm = layer_norm
        assert not (operation == "single" and fm_norm)  # unstable

    def forward(self, fourmomenta, scalars=None, ptr=None, **kwargs):
        attn_kwargs = {}
        if ptr is not None:
            batch = get_batch_from_ptr(ptr)
            on_cpu = fourmomenta.device == torch.device("cpu")
            mask = get_xformers_attention_mask(
                batch, materialize=on_cpu, dtype=scalars.dtype
            )
            attn_kwargs["attn_mask" if on_cpu else "attn_bias"] = mask
            fourmomenta = fourmomenta.unsqueeze(0)
            scalars = scalars.unsqueeze(0)

        # get query and key from LGATr
        mv = embed_vector(fourmomenta).unsqueeze(-2).to(scalars.dtype)
        qk_mv, qk_s = self.net(mv, scalars, **attn_kwargs)
        if self.lgatr_norm is not None:
            qk_mv, qk_s = self.lgatr_norm(qk_mv, qk_s)

        # flatten for message passing
        edge_index, batch, ptr = get_edge_index_and_batch(
            fourmomenta, ptr, remove_self_loops=False
        )
        in_shape = fourmomenta.shape[:-1]
        fourmomenta = fourmomenta.reshape(math.prod(in_shape), 4)
        qk_mv = qk_mv.reshape(math.prod(in_shape), qk_mv.shape[-2], qk_mv.shape[-1])
        qk_s = qk_s.reshape(math.prod(in_shape), qk_s.shape[-1])

        # extract q and k
        q_mv, k_mv = torch.chunk(qk_mv.to(fourmomenta.dtype), chunks=2, dim=-2)
        q_s, k_s = torch.chunk(qk_s.to(fourmomenta.dtype), chunks=2, dim=-1)

        # unpack the n_vectors axis
        q_mv = q_mv.reshape(*q_mv.shape[:-2], self.n_vectors, -1, q_mv.shape[-1])
        k_mv = k_mv.reshape(*k_mv.shape[:-2], self.n_vectors, -1, k_mv.shape[-1])
        q_s = q_s.reshape(*q_s.shape[:-1], self.n_vectors, -1)
        k_s = k_s.reshape(*k_s.shape[:-1], self.n_vectors, -1)

        qk_product = get_qk_product(q_mv, k_mv, q_s, k_s, edge_index)

        # message-passing
        vecs = self.propagate(
            edge_index,
            fm=fourmomenta,
            prefactor=qk_product,
            batch=batch,
            node_ptr=ptr,
        )
        vecs = vecs.reshape(fourmomenta.shape[0], -1, 4)

        if self.layer_norm:
            norm = lorentz_squarednorm(vecs).sum(dim=-1, keepdim=True).unsqueeze(-1)
            vecs = vecs / norm.abs().sqrt().clamp(min=1e-5)

        # reshape result
        vecs = vecs.reshape(*in_shape, -1, 4)
        return vecs

    def message(
        self,
        edge_index,
        fm_i,
        fm_j,
        node_ptr,
        batch,
        prefactor,
    ):
        # prepare fourmomenta
        fm_rel = self.operation(fm_i, fm_j)
        if self.fm_norm:
            fm_rel_norm = lorentz_squarednorm(fm_rel).unsqueeze(-1)
            fm_rel_norm = fm_rel_norm.abs().sqrt().clamp(min=1e-6)
        else:
            fm_rel_norm = 1.0
        fm_rel = (fm_rel / fm_rel_norm)[:, None, :4]

        prefactor = self.nonlinearity(
            prefactor,
            index=edge_index[0],
            node_ptr=node_ptr,
            node_batch=batch,
            remove_self_loops=False,
        )
        prefactor = prefactor.unsqueeze(-1)
        out = prefactor * fm_rel
        out = out.reshape(out.shape[0], -1)
        return out


def get_qk_product(q_mv, k_mv, q_s, k_s, edge_index):
    # prepare queries and keys
    q = torch.cat(
        [
            rearrange(
                q_mv
                * _load_inner_product_factors(device=q_mv.device, dtype=q_mv.dtype),
                "... c x -> ... (c x)",
            ),
            q_s,
        ],
        -1,
    )
    k = torch.cat([rearrange(k_mv, "... c x -> ... (c x)"), k_s], -1)

    # evaluate attention weights on edges
    scale_factor = 1 / math.sqrt(q.shape[-1])
    src, dst = edge_index
    q_edges, k_edges = q[dst], k[src]
    qk_product = (q_edges * k_edges).sum(dim=-1) * scale_factor
    return qk_product
