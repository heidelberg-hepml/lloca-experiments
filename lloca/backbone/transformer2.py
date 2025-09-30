from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import LLoCaAttention
from ..reps.tensorreps import TensorReps

EPS_NORM = 1e-5


class MultiHeadQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-head attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    **factory_kwargs
    """

    def __init__(self, in_channels, hidden_channels, num_heads, **factory_kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(
            in_channels, 3 * hidden_channels * num_heads, **factory_kwargs
        )

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.linear.weight.chunk(3, dim=0):
            nn.init.xavier_uniform_(w)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        """Forward pass.

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        qkv = self.linear(inputs)  # (..., num_items, 3 * hidden_channels * num_heads)

        *leading, items, last = qkv.shape
        hidden_channels = last // (3 * self.num_heads)
        qkv = qkv.view(*leading, items, 3, hidden_channels, self.num_heads)
        qkv = qkv.movedim(-3, 0).movedim(-1, len(leading) + 1)
        q, k, v = qkv.unbind(dim=0)  # 3x (..., num_heads, num_items, hidden_channels)
        return q, k, v


class SelfAttention(nn.Module):
    """Baseline self-attention layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    attention
    num_heads : int
        Number of attention heads.
    dropout_prob : float
        Dropout probability for output.
    **factory_kwargs
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        attention,
        num_heads: int = 8,
        dropout_prob=None,
        **factory_kwargs,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        self.attention = attention

        # Linear maps
        self.qkv_linear = MultiHeadQKVLinear(
            in_channels, hidden_channels, num_heads, **factory_kwargs
        )
        self.out_linear = nn.Linear(
            hidden_channels * num_heads, out_channels, **factory_kwargs
        )

        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.out_linear.weight, gain=0.1)
        if self.out_linear.bias is not None:
            nn.init.zeros_(self.out_linear.bias)

    def forward(self, inputs: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs
        """
        q, k, v = self.qkv_linear(
            inputs
        )  # each: (..., num_heads, num_items, num_channels)

        # Attention layer
        h = self.attention(
            q.contiguous(),
            k.expand_as(q).contiguous(),
            v.expand_as(q),
            **attn_kwargs,
        )

        # Concatenate heads and transform linearly
        *leading, num_heads, num_items, hidden_channels = h.shape
        h = h.permute(*range(len(leading)), -2, -3, -1)
        h = h.reshape(*leading, num_items, num_heads * hidden_channels)

        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs


class TransformerBlock(nn.Module):
    """Baseline transformer block.

    Inputs are first processed by a block consisting of RMSNorm, multi-head self-attention, and
    residual connection. Then the data is processed by a block consisting of another RMSNorm, an
    item-wise two-layer MLP with SwiGLU activations, and another residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    attention
    num_heads : int
        Number of attention heads.
    dropout_prob : float
        Dropout probability for output.
    **factory_kwargs
    """

    def __init__(
        self,
        channels,
        attention,
        num_heads: int,
        dropout_prob=None,
        **factory_kwargs,
    ) -> None:
        super().__init__()

        self.norm1 = nn.RMSNorm(channels, eps=EPS_NORM)
        self.norm2 = nn.RMSNorm(channels, eps=EPS_NORM)
        self.norm3 = nn.RMSNorm(channels, eps=EPS_NORM)
        self.norm4 = nn.RMSNorm(4 * channels, eps=EPS_NORM)

        self.attention = SelfAttention(
            channels,
            channels,
            channels // num_heads,
            attention,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            **factory_kwargs,
        )

        self.linear_in = nn.Linear(channels, 4 * channels, **factory_kwargs)
        self.dropout = (
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.linear_out = nn.Linear(2 * channels, channels, **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.linear_in.weight.chunk(2, dim=0):
            nn.init.xavier_uniform_(w)
        if self.linear_in.bias is not None:
            nn.init.zeros_(self.linear_in.bias)

        nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

    def forward(self, inputs: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs
        """

        # Residual attention
        h = self.norm1(inputs)  # standard pre-LN
        h = self.attention(h, **attn_kwargs)
        h = self.norm3(h)  # extra norm in NormFormer
        outputs = inputs + h

        # Residual MLP
        h = self.norm2(outputs)  # standard pre-LN
        h = self.linear_in(h)
        h = self.norm4(h)  # extra norm in NormFormer
        a, b = h.chunk(2, dim=-1)
        h = self.activation(a) * b  # SwiGLU activation
        h = self.linear_out(h)
        h = self.dropout(h)
        outputs = outputs + h

        return outputs


class Transformer(nn.Module):
    """Baseline transformer.

    Combines num_blocks transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    attn_reps : str
        Representation of each attention head.
    out_channels : int
        Number of output channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    checkpoint_blocks : bool
        Use gradient checkpointing for transformer blocks.
    dropout_prob : float
        Dropout probability for output.
    """

    def __init__(
        self,
        in_channels: int,
        attn_reps: str,
        out_channels: int,
        num_blocks: int,
        num_heads: int,
        checkpoint_blocks: bool = False,
        dropout_prob=None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        attn_reps = TensorReps(attn_reps)
        self.hidden_channels = attn_reps.dim * num_heads
        self.checkpoint_blocks = checkpoint_blocks
        self.attention = LLoCaAttention(attn_reps, num_heads)
        factory_kwargs = {"bias": use_bias}

        self.linear_in = nn.Linear(in_channels, self.hidden_channels, **factory_kwargs)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden_channels,
                    attention=self.attention,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                    **factory_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(
            self.hidden_channels, out_channels, **factory_kwargs
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_in.weight)
        if self.linear_in.bias is not None:
            nn.init.zeros_(self.linear_in.bias)

        nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

    def forward(self, inputs: torch.Tensor, frames, **attn_kwargs) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data with shape (..., num_items, in_channels)
        frames : Frames
            Local frames used for invariant particle attention
        **attn_kwargs

        Returns
        -------
        outputs : Tensor
            Outputs with shape (..., num_items, out_channels)
        """
        self.attention.prepare_frames(frames)

        h = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                fn = partial(block, **attn_kwargs)
                h = checkpoint(fn, h)
            else:
                h = block(h, **attn_kwargs)
        outputs = self.linear_out(h)
        return outputs
