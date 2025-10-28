import torch
from lgatr import embed_vector, extract_vector
from lgatr.primitives.attention import sdp_attention
from lgatr.layers import EquiLayerNorm

from .base import EquiVectors
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
        compensate=False,
        norm1=False,
        norm2=False,
    ):
        super().__init__()
        self.n_vectors = n_vectors
        if compensate:
            hidden_mv_channels *= 2 * n_vectors
            hidden_s_channels *= 2 * n_vectors
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
        self.norm1 = norm1
        self.norm2 = norm2

        self.norm = EquiLayerNorm()

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
        if self.norm1:
            out_mv, out_s = self.norm(out_mv, out_s)

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
        if self.norm2:
            out_mv, out_s = self.norm(out_mv, out_s)
        out_mv = out_mv.transpose(-3, -4)

        # extract vector part of multivector output
        out = extract_vector(out_mv).squeeze(-2)

        if ptr is not None:
            out = out.squeeze(0)  # undo initial unsqueeze(0)
        return out
