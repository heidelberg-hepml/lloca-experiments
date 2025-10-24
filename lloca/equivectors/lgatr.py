import torch
from lgatr import embed_vector, extract_vector
from lgatr.primitives.attention import sdp_attention

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

    def init_standardization(self, fourmomenta, ptr=None):
        pass

    def forward(self, fourmomenta, scalars=None, ptr=None):
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
        q_s, k_s = torch.chunk(out_s.to(fourmomenta.dtype), chunks=2, dim=-1)
        q_mv, k_mv = torch.chunk(out_mv.to(fourmomenta.dtype), chunks=2, dim=-2)
        qs_s = torch.chunk(q_s, chunks=self.n_vectors, dim=-1)
        ks_s = torch.chunk(k_s, chunks=self.n_vectors, dim=-1)
        qs_mv = torch.chunk(q_mv, chunks=self.n_vectors, dim=-2)
        ks_mv = torch.chunk(k_mv, chunks=self.n_vectors, dim=-2)

        # attention and final reshape
        v_mv = embed_vector(fourmomenta).unsqueeze(-2)
        v_s = torch.zeros_like(v_mv[..., [], 0])
        out = []
        for q_s, k_s, q_mv, k_mv in zip(qs_s, ks_s, qs_mv, ks_mv):
            out_mv, _ = sdp_attention(
                q_mv=q_mv,
                k_mv=k_mv,
                q_s=q_s,
                k_s=k_s,
                v_mv=v_mv,
                v_s=v_s,
                **attn_kwargs
            )
            out_v = extract_vector(out_mv)
            out.append(out_v)
        out = torch.stack(out, dim=-2)
        if ptr is not None:
            out = out.squeeze(0)
        return out
