"""xformers memory-efficient attention backend."""
import torch

try:
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
except ModuleNotFoundError:
    raise ImportError(
        "xformers is not installed. Run 'pip install lloca[xformers_attention]'."
    )


def attention(query, key, value, **kwargs):
    """Pass to xformers memory-efficient attention.
    Note that this xformers expects the shape (batch, head, items_out, channel).

    Parameters
    ----------
    query : torch.Tensor
        Queries with shape (batch, head, items_out, channel)
    key : torch.Tensor
        Keys with shape (batch, head, items_in, channel)
    value : torch.Tensor
        Values with shape (batch, head, items_in, channel)
    **kwargs
        Additional keyword arguments passed to memory_efficient_attention.

    Returns
    -------
    out : torch.Tensor
        Result with shape (batch, head, items_out, channel)
    """
    assert (
        len(query.shape) == 4
    ), "xformers constrains attention input shape to (batch, head, items, channel)."
    if key.shape[1] != query.shape[1]:
        # manual broadcasting for key and value; required for multi-query attention
        key = key.expand(key.shape[0], query.shape[1], *key.shape[2:])
        value = value.expand(value.shape[0], query.shape[1], *value.shape[2:])

    # xformers expects input shape (batch, item, head, channel)
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    out = memory_efficient_attention(query, key, value, **kwargs)
    out = out.transpose(1, 2).contiguous()
    return out


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
