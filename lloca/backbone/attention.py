"""LLoCa attention module."""
from math import prod
import torch
from torch import Tensor

from ..framesnet.frames import Frames, InverseFrames, LowerIndicesFrames
from ..reps.tensorreps import TensorReps
from ..reps.tensorreps_transform import TensorRepsTransform
from .attention_backends import get_attention_backend


class LLoCaAttention(torch.nn.Module):
    def __init__(self, attn_reps, num_heads):
        """Attention with frame-to-frame transformations.

        Parameters
        ----------
        attn_reps : TensorReps
            Tensor representation of a single attention head.
        num_heads : int
            Number of attention heads
        """
        super().__init__()
        self.transform = TensorRepsTransform(TensorReps(attn_reps))
        self.num_heads = num_heads

        self.frames = None
        self.inv_frames = None
        self.lower_inv_frames = None

    def prepare_frames(self, frames):
        """Prepare local frames for processing with LLoCa attention.
        For a single forward pass through the network, this method is
        called only once for efficiency.

        Parameters
        ----------
        frames: torch.tensor
            Local frames of reference for each particle of shape (..., N, 4, 4)
            where N is the number of particles.
        """
        self.frames = frames
        if not self.frames.is_global:
            # insert frames head dimension
            self.frames = self.frames.reshape(
                *frames.shape[:-3], 1, frames.shape[-3], 4, 4
            )
            self.frames = self.frames.repeat(
                *((1,) * len(frames.shape[:-3])), self.num_heads, 1, 1, 1
            )

            # create inv_frames and lower_inv_frames
            inv_frames = InverseFrames(self.frames)
            lower_inv_frames = LowerIndicesFrames(inv_frames)

            # qkv = (inv_frames, lower_inv_frames, inv_frames)
            # note that (lower_inv_frames, inv_frames, inv_frames) is equivalent
            self.frames_qkv = Frames(
                matrices=torch.stack(
                    [
                        inv_frames.matrices,
                        lower_inv_frames.matrices,
                        inv_frames.matrices,
                    ],
                    dim=0,
                ),
                is_identity=inv_frames.is_identity,
                is_global=inv_frames.is_global,
                det=torch.stack(
                    [inv_frames.det, lower_inv_frames.det, inv_frames.det], dim=0
                ),
                inv=torch.stack(
                    [inv_frames.inv, lower_inv_frames.inv, inv_frames.inv], dim=0
                ),
            )

            # flatten frames (preparation for tensorreps_transform)
            self.frames = self.frames.reshape(-1, 4, 4)
            self.frames_qkv = self.frames_qkv.reshape(-1, 4, 4)

    def forward(self, q_local, k_local, v_local, **attn_kwargs):
        """Execute LLoCa attention.

        Strategy
        1) Transform q, k, v into global frame
        2) Apply attention in global frame
        3) Transform output back into local frame

        Comments
        - Dimensions: ... (optional), H (head), N (particles), C (channels).
        - Extension to cross-attention is trivial but we don't have this right now for convenience. Strategy: frames_q for queries (in contrast to frames=frames_kv).

        Parameters
        ----------
        q_local: torch.tensor
            Local queries of shape (..., H, N, C)
        k_local: torch.tensor
            Local keys of shape (..., H, N, C)
        v_local: torch.tensor
            Local values of shape (..., H, N, C)
        **attn_kwargs
            Optional arguments that are passed on to the attention backend

        Returns
        -------
        out_local: torch.tensor
            Attention output in local frame of shape (..., H, N, C)
        """
        if self.frames.is_global:
            # fallback to standard attention for global frames
            return scaled_dot_product_attention(
                q_local,
                k_local,
                v_local,
                **attn_kwargs,
            )

        # check input shapes
        assert k_local.shape == v_local.shape == q_local.shape  # has to match perfectly
        assert 3 * prod(k_local.shape[:-1]) == self.frames_qkv.shape[-3]

        # transform q, k, v into global frame
        qkv_local = torch.stack([q_local, k_local, v_local], dim=0)
        qkv_global = self.transform(qkv_local, self.frames_qkv)
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

        out_global = out_global.view(*shape_q)  # (..., H, N, C)

        # transform result back into local frame
        out_local = self.transform(out_global, self.frames)
        return out_local


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    **attn_kwargs,
) -> Tensor:
    """Execute scaled dot-product attention.
    The attention backend is determined dynamically
    based on the ``**attn_kwargs``.

    Parameters
    ----------
    query : torch.Tensor
        Tensor of shape (..., items_out, channels)
    key : torch.Tensor
        Tensor of shape (..., items_in, channels)
    value : torch.Tensor
        Tensor of shape (..., items_in, channels)
    **attn_kwargs
        Optional keyword arguments passed to attention.

    Returns
    -------
    torch.Tensor
        Tensor of shape (..., head, item_out, channels)
    """
    attention_backend = get_attention_backend(**attn_kwargs)
    return attention_backend(query, key, value, **attn_kwargs)
