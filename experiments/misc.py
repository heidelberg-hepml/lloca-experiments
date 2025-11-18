from collections.abc import Mapping

import torch
from xformers.ops.fmha.attn_bias import BlockDiagonalMask


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary with str keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
    mask = BlockDiagonalMask.from_seqlens(bincounts, device=batch.device)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device, dtype=dtype)
    return mask
