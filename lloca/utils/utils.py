"""Tools for graph construction and manipulation."""

import torch

from .lorentz import lorentz_squarednorm


def get_batch_from_ptr(ptr):
    """Reconstruct batch indices (batch) from pointer (ptr).

    Parameters
    ----------
    ptr : torch.Tensor
        Pointer tensor indicating the start of each batch.
        Tensor of shape (B+1,) where B is the number of batches.

    Returns
    -------
    torch.Tensor
        A tensor where each element indicates the batch index for each item.
        Tensor of shape (N,) where N is the total number of items across all batches.
    """
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1],
    )


def get_ptr_from_batch(batch):
    """Reconstruct pointer (ptr) from batch indices (batch).

    Parameters
    ----------
    batch : torch.Tensor
        A tensor where each element indicates the batch index for each item.
        Tensor of shape (N,) where N is the total number of items across all batches.

    Returns
    -------
    torch.Tensor
        A pointer tensor indicating the start of each batch.
        Tensor of shape (B+1,) where B is the number of batches.
    """
    return torch.cat(
        [
            torch.tensor([0], device=batch.device),
            torch.where(batch[1:] - batch[:-1] != 0)[0] + 1,
            torch.tensor([batch.shape[0]], device=batch.device),
        ],
        0,
    )


def get_edge_index_from_ptr(ptr, remove_self_loops=True):
    """Construct edge index of fully connected graph from pointer (ptr).
    This function should be used for graphs represented by sparse tensors,
    i.e. graphs where the number of nodes per graph can vary.

    Parameters
    ----------
    ptr : torch.Tensor
        Pointer tensor indicating the start of each batch.
        Tensor of shape (B+1,) where B is the number of batches.
    remove_self_loops : bool, optional
        Whether to remove self-loops from the edge index, by default True.

    Returns
    -------
    torch.Tensor
        A tensor of shape (2, E) where E is the number of edges, representing the edge index.
    """
    row = torch.arange(ptr.max(), device=ptr.device)
    diff = ptr[1:] - ptr[:-1]
    repeats = (diff).repeat_interleave(diff)
    row = row.repeat_interleave(repeats)

    repeater = torch.stack(
        (-diff + 1, torch.ones_like(diff, device=ptr.device))
    ).T.reshape(-1)
    extras = repeater.repeat_interleave(repeater.abs())
    integ = torch.ones(row.shape[0], dtype=torch.long, device=ptr.device)
    mask = (row[1:] - row[:-1]).to(torch.bool)
    integ[0] = 0
    integ[1:][mask] = extras[:-1]
    col = torch.cumsum(integ, 0)

    edge_index = torch.stack((row, col))

    if remove_self_loops:
        row, col = edge_index
        edge_index = edge_index[:, row != col]

    return edge_index


def get_edge_index_from_shape(features_ref, remove_self_loops=True):
    """Construct edge index of fully connected graph from reference object.
    Only shape and device of the reference object are used.
    This function should be used for graphs represented by dense tensors,
    i.e. graphs where the number of nodes per graph is fixed.

    Parameters
    ----------
    features_ref : torch.Tensor
        Reference tensor from which the shape and device are derived.
    remove_self_loops : bool, optional
        Whether to remove self-loops from the edge index, by default True.

    Returns
    -------
    torch.Tensor
        A tensor of shape (2, E) where E is the number of edges, representing the edge index.
    """
    B, N, _ = features_ref.shape
    device = features_ref.device

    nodes = torch.arange(N, device=device)
    row = nodes.repeat_interleave(N)
    col = nodes.repeat(N)

    if remove_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]

    edge_base = torch.stack([row, col], dim=0)

    offsets = torch.arange(B, device=device, dtype=torch.long) * N
    batched = edge_base.unsqueeze(2) + offsets.view(1, 1, -1)
    edge_index_global = batched.permute(0, 2, 1).reshape(2, -1)

    batch = torch.arange(B, device=device).repeat_interleave(N)
    return edge_index_global, batch


def get_edge_attr(fourmomenta, edge_index, eps=1e-10, use_float64=True):
    """Calculate edge attributes based on the squared Lorentz norm of the sum of four-momenta.

    Parameters
    ----------
    fourmomenta : torch.Tensor
        A tensor of shape (B, N, 4) representing the four-momenta of particles.
    edge_index : torch.Tensor
        A tensor of shape (2, E) representing the edge index of the graph.
    eps : float, optional
        A small value to avoid log(0) issues, by default 1e-10.
    use_float64 : bool, optional
        Whether to use float64 precision for calculations, by default True.

    Returns
    -------
    torch.Tensor
        A tensor of shape (E,) representing the edge attributes, which are the logarithm of the squared Lorentz norm.
    """
    if use_float64:
        in_dtype = fourmomenta.dtype
        fourmomenta = fourmomenta.to(torch.float64)
    mij2 = lorentz_squarednorm(fourmomenta[edge_index[0]] + fourmomenta[edge_index[1]])
    edge_attr = mij2.clamp(min=eps).log()
    if use_float64:
        edge_attr = edge_attr.to(in_dtype)
    return edge_attr
