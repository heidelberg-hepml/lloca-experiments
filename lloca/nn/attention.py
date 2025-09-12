from typing import Optional, Union

from math import prod
import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
from xformers.ops import AttentionBias, memory_efficient_attention
from xformers.ops.fmha import BlockDiagonalMask

from ..lframes.lframes import (
    LFrames,
    InverseLFrames,
    LowerIndices,
)
from ..reps.tensorreps import TensorReps
from ..reps.tensorreps_transform import TensorRepsTransform


class LLoCaAttention(torch.nn.Module):
    """"""

    def __init__(self, attn_reps, num_heads):
        super().__init__()
        self.transform = TensorRepsTransform(TensorReps(attn_reps))
        self.num_heads = num_heads

        self.lframes = None
        self.inv_lframes = None
        self.lower_inv_lframes = None

    def prepare_lframes(self, lframes):
        """Prepare local frames for processing with LLoCa attention.
        For a single forward pass through the network, this method is
        called only once for efficiency.

        Parameters
        ----------
        lframes: torch.tensor
            Local frames of reference for each particle of shape (..., N, 4, 4)
            where N is the number of particles.
        """
        self.lframes = lframes
        if not self.lframes.is_global:
            # insert lframes head dimension
            self.lframes = self.lframes.reshape(
                *lframes.shape[:-3], 1, lframes.shape[-3], 4, 4
            )
            self.lframes = self.lframes.repeat(
                *((1,) * len(lframes.shape[:-3])), self.num_heads, 1, 1, 1
            )

            # create inv_lframes and lower_inv_lframes
            inv_lframes = InverseLFrames(self.lframes)
            lower_inv_lframes = LowerIndices(inv_lframes)

            # qkv = (inv_lframes, lower_inv_lframes, inv_lframes)
            # note that (lower_inv_lframes, inv_lframes, inv_lframes) is equivalent
            self.lframes_qkv = LFrames(
                matrices=torch.stack(
                    [
                        inv_lframes.matrices,
                        lower_inv_lframes.matrices,
                        inv_lframes.matrices,
                    ],
                    dim=0,
                ),
                is_identity=inv_lframes.is_identity,
                is_global=inv_lframes.is_global,
                det=torch.stack(
                    [inv_lframes.det, lower_inv_lframes.det, inv_lframes.det], dim=0
                ),
                inv=torch.stack(
                    [inv_lframes.inv, lower_inv_lframes.inv, inv_lframes.inv], dim=0
                ),
            )

            # flatten lframes (preparation for tensorreps_transform)
            self.lframes = self.lframes.reshape(-1, 4, 4)
            self.lframes_qkv = self.lframes_qkv.reshape(-1, 4, 4)

    def forward(self, q_local, k_local, v_local, **attn_kwargs):
        """Execute LLoCa attention.

        Strategy
        1) Transform q, k, v into global frame
        2) Apply attention in global frame
        3) Transform output back into local frame

        Comments
        - dimensions: *dims (optional), H (head), N (particles), C (channels)
        - extension to cross-attention is trivial but we don't have this right now for convenience
          strategy: lframes_q for queries (in contrast to lframes=lframes_kv)

        Parameters
        ----------
        q_local: torch.tensor
            Local queries of shape (*dims, H, N, C)
        k_local: torch.tensor
            Local keys of shape (*dims, H, N, C)
        v_local: torch.tensor
            Local values of shape (*dims, H, N, C)
        attn_kwargs: dict
            Optional arguments that are passed on to attention

        Returns
        -------
        out_local: torch.tensor
            Attention output in local frame of shape (*dims, H, N, C)
        """
        if self.lframes.is_global:
            # shortcut if global_frame = local_frame
            return scaled_dot_product_attention(
                q_local,
                k_local,
                v_local,
                **attn_kwargs,
            )

        # check input shapes
        assert k_local.shape == v_local.shape == q_local.shape  # has to match perfectly
        assert 3 * prod(k_local.shape[:-1]) == self.lframes_qkv.shape[-3]

        qkv_local = torch.stack([q_local, k_local, v_local], dim=0)
        qkv_global = self.transform(qkv_local, self.lframes_qkv)
        q_global, k_global, v_global = torch.unbind(qkv_global, dim=0)

        # (B, H, N, C) format required for scaled_dot_product_attention
        shape_q, shape_k = q_global.shape, k_global.shape
        q_global = q_global.reshape(-1, *shape_q[-3:])
        k_global = k_global.reshape(-1, *shape_k[-3:])
        v_global = v_global.reshape(-1, *shape_k[-3:])

        # attention (in global frame)
        out_global = scaled_dot_product_attention(
            q_global,
            k_global,
            v_global,
            **attn_kwargs,
        )

        out_global = out_global.view(*shape_q)  # (*dims, H, N, C)

        # transform out back into local frame
        out_local = self.transform(out_global, self.lframes)
        return out_local


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Union[AttentionBias, Tensor]] = None,
    is_causal=False,
    dropout_p=0.0,
) -> Tensor:
    """Execute (vanilla) scaled dot-product attention.

    Dynamically dispatch to xFormers if attn_mask is an instance of xformers.ops.AttentionBias

    Parameters
    ----------
    query : Tensor
        of shape [batch, head, item, d]
    key : Tensor
        of shape [batch, head, item, d]
    value : Tensor
        of shape [batch, head, item, d]
    attn_mask : Optional[Union[AttentionBias, Tensor]]
        Attention mask
    is_causal: bool

    Returns
    -------
    Tensor
        of shape [batch, head, item, d]
    """
    if isinstance(attn_mask, AttentionBias):
        in_dtype = query.dtype
        query, key, value = (
            query.to(torch.float32),
            key.to(torch.float32),
            value.to(torch.float32),
        )
        assert (
            not is_causal
        ), "is_causal=True not implemented yet for xformers attention"
        if key.shape[1] != query.shape[1]:  # required to make multi_query work
            key = key.expand(key.shape[0], query.shape[1], *key.shape[2:])
            value = value.expand(value.shape[0], query.shape[1], *value.shape[2:])
        query = query.transpose(
            1, 2
        )  # [batch, head, item, d] -> [batch, item, head, d]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = memory_efficient_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_bias=attn_mask,
        )
        out = out.transpose(1, 2)  # [batch, item, head, d] -> [batch, head, item, d]
        return out.to(in_dtype)
    return torch_sdpa(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
    )


def get_xformers_attention_mask(batch, materialize=False, dtype=torch.float32):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch.tensor
        batch object in the torch_geometric.data naming convention
        contains batch index for each event in a sparse tensor
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(
            batch.device, dtype=dtype
        )
    return mask
