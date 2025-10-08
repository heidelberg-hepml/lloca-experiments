import torch


def generate_batch(G=8, N_range=[10, 20], C=16, batch=None, edge_index=None):
    """
    Generate random batch, edge_index, graph, nodes, and edges for testing,
    assuming a fully connected graph.

    Parameters
    ----------
    G : int
        Number of graphs in the batch.
    N_range : list
        Range of number of nodes per graph.
    C : int
        Number of channels (features) for graph, nodes, and edges.
    batch : torch.Tensor, optional
        Predefined batch tensor, by default None.
    edge_index : torch.Tensor, optional
        Predefined edge_index tensor, by default None.

    Returns
    -------
    batch : torch.Tensor
        Batch tensor of shape (N).
    edge_index : torch.Tensor
        Edge index tensor of shape (2, E).
    graph : torch.Tensor
        Graph-level features of shape (G, C).
    nodes : torch.Tensor
        Node-level features of shape (G, N).
    edges : torch.Tensor
        Edge-level features of shape (G, E).
    """
    if batch is None or edge_index is None:
        length = torch.randint(low=N_range[0], high=N_range[1], size=(G,))
        ptr = torch.zeros(G + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(length, dim=0)
        batch = get_batch_from_ptr(ptr)
        edge_index = get_edge_index_from_ptr(ptr, remove_self_loops=False)
    else:
        assert batch is not None and edge_index is not None
    graph = torch.randn(G, C) if C > 0 else None
    nodes = torch.randn(batch.numel(), C) if C > 0 else None
    edges = torch.randn(edge_index.size(1), C) if C > 0 else None
    return batch, edge_index, graph, nodes, edges


def get_batch_from_ptr(ptr):
    """Reconstruct batch indices (batch) from pointer (ptr).

    Parameters
    ----------
    ptr : torch.Tensor
        Pointer tensor indicating the start of each batch.

    Returns
    -------
    torch.Tensor
        A tensor where each element indicates the batch index for each item.
    """
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1],
    )


def get_edge_index_from_ptr(ptr, remove_self_loops=True):
    """Construct edge index of fully connected graph from pointer (ptr).

    Parameters
    ----------
    ptr : torch.Tensor
        Pointer tensor indicating the start of each batch.
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
