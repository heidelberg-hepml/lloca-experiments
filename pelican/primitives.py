import torch


def bell_number(n):
    """Compute the Bell number for small n (0 <= n <= 4).
    The Bell number is the number of ways to partition a set of n elements.
    In PELICAN, it corresponds to the number of aggregation maps
    for a given rank n = in_rank + out_rank.
    """
    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    elif n == 3:
        return 5
    elif n == 4:
        return 15
    else:
        raise NotImplementedError(
            f"Asking for bell_number(n={n}), but bell_number(n>4) not implemented yet."
        )


def aggregate_0to2(graph, edge_index, batch, **kwargs):
    """Aggregate from graph (rank 0) to edges (rank 2).

    Parameters
    ----------
    graph : torch.Tensor
        Graph-level features of shape (G, C).
    edge_index : torch.Tensor
        Edge index tensor of shape (2, E).
    batch : torch.Tensor
        Batch tensor of shape (N).
    **kwargs

    Returns
    -------
    ops : torch.Tensor
        Aggregated features of shape (E, C, 2).
    """
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(graph)

    ops = torch.stack(
        [
            graph[edge_batch],
            graph[edge_batch] * diag_mask,
        ],
        dim=-1,
    )  # shape (E, C, 2)
    return ops


def aggregate_1to2(nodes, edge_index, batch, reduce="mean", **kwargs):
    """Aggregate from nodes (rank 1) to edges (rank 2).

    Parameters
    ----------
    nodes : torch.Tensor
        Node-level features of shape (N, C).
    edge_index : torch.Tensor
        Edge index tensor of shape (2, E).
    batch : torch.Tensor
        Batch tensor of shape (N).
    reduce : str, optional
        Reduction method to use ('mean', 'sum', 'prod', 'amin', 'amax'), by default 'mean'.
    **kwargs

    Returns
    -------
    ops : torch.Tensor
        Aggregated features of shape (E, C, 5).
    """
    _, C = nodes.shape
    E = edge_index.size(1)
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(nodes)

    nodes_agg = custom_scatter(nodes, batch, dim_size=E, C=C, reduce=reduce)
    is_diag = is_diag.unsqueeze(-1)

    ops = torch.stack(
        [
            nodes[row] * diag_mask,
            nodes[row],
            nodes[col],
            nodes_agg[edge_batch] * diag_mask,
            nodes_agg[edge_batch],
        ],
        dim=-1,
    )  # shape (E, C, 5)
    return ops


def aggregate_2to0(edges, edge_index, batch, reduce="mean", G=None, **kwargs):
    """Aggregate from edges (rank 2) to graph (rank 0).

    Parameters
    ----------
    edges : torch.Tensor
        Edge-level features of shape (E, C).
    edge_index : torch.Tensor
        Edge index tensor of shape (2, E).
    batch : torch.Tensor
        Batch tensor of shape (N).
    reduce : str, optional
        Reduction method to use ('mean', 'sum', 'prod', 'amin', 'amax'), by default 'mean'.
    G : int, optional
        Number of graphs in the batch. If None, it will be inferred from the batch tensor.
    **kwargs

    Returns
    -------
    ops : torch.Tensor
        Aggregated features of shape (G, C, 2).
    """
    _, C = edges.shape
    if G is None:
        # host synchronization causes slowdown; maybe there is a better way?
        G = batch[-1].item() + 1
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags = edges * diag_mask
    graph_agg = custom_scatter(edges, edge_batch, dim_size=G, C=C, reduce=reduce)
    diag_agg = custom_scatter(diags, edge_batch, dim_size=G, C=C, reduce=reduce)

    ops = torch.stack(
        [
            graph_agg,
            diag_agg,
        ],
        dim=-1,
    )  # shape (G, C, 2)
    return ops


def aggregate_2to1(edges, edge_index, batch, reduce="mean", **kwargs):
    """Aggregate from edges (rank 2) to nodes (rank 1).

    Parameters
    ----------
    edges : torch.Tensor
        Edge-level features of shape (E, C).
    edge_index : torch.Tensor
        Edge index tensor of shape (2, E).
    batch : torch.Tensor
        Batch tensor of shape (N).
    reduce : str, optional
        Reduction method to use ('mean', 'sum', 'prod', 'amin', 'amax'), by default 'mean'.
    **kwargs

    Returns
    -------
    ops : torch.Tensor
        Aggregated features of shape (N, C, 5).
    """
    _, C = edges.shape
    N = batch.size(0)
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags_padded = edges * diag_mask
    diags = custom_scatter(diags_padded, row, dim_size=N, C=C, reduce="sum")
    row_agg = custom_scatter(edges, row, dim_size=N, C=C, reduce=reduce)
    col_agg = custom_scatter(edges, col, dim_size=N, C=C, reduce=reduce)
    graph_agg = custom_scatter(edges, edge_batch, dim_size=N, C=C, reduce=reduce)
    diag_agg = custom_scatter(diags_padded, edge_batch, dim_size=N, C=C, reduce=reduce)

    ops = torch.stack(
        [
            diags,
            row_agg,
            col_agg,
            graph_agg[batch],
            diag_agg[batch],
        ],
        dim=-1,
    )  # shape (N, C, 5)
    return ops


def aggregate_2to2(edges, edge_index, batch, reduce="mean", **kwargs):
    """Aggregate from edges (rank 2) to edges (rank 2).

    Parameters
    ----------
    edges : torch.Tensor
        Edge-level features of shape (E, C).
    edge_index : torch.Tensor
         Edge index tensor of shape (2, E).
    batch : torch.Tensor
        Batch tensor of shape (N).
    reduce : str, optional
        Reduction method to use ('mean', 'sum', 'prod', 'amin', 'amax'), by default 'mean'.
    **kwargs

    Returns
    -------
    ops : torch.Tensor
        Aggregated features of shape (E, C, 15).
    """
    _, C = edges.shape
    N = batch.size(0)
    row, col = edge_index
    perm_T = get_transpose(edge_index)
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags_padded = edges * diag_mask
    diags = custom_scatter(diags_padded, row, dim_size=N, C=C, reduce="sum")
    row_agg = custom_scatter(edges, row, dim_size=N, C=C, reduce=reduce)
    col_agg = custom_scatter(edges, col, dim_size=N, C=C, reduce=reduce)
    graph_agg = custom_scatter(edges, edge_batch, dim_size=N, C=C, reduce=reduce)
    diag_agg = custom_scatter(diags_padded, edge_batch, dim_size=N, C=C, reduce=reduce)

    # Note: creating ops as empty tensor and filling it is ~4% faster but requires ~10% more RAM
    ops = torch.stack(
        [
            edges,
            edges[perm_T],
            diags_padded,
            diags[row],
            diags[col],
            col_agg[row] * diag_mask,
            row_agg[col] * diag_mask,
            col_agg[row],
            col_agg[col],
            row_agg[row],
            row_agg[col],
            diag_agg[edge_batch],
            diag_agg[edge_batch] * diag_mask,
            graph_agg[edge_batch],
            graph_agg[edge_batch] * diag_mask,
        ],
        dim=-1,
    )  # shape (E, C, 15)
    return ops


def get_transpose(edge_index):
    """Get the permutation that transposes the edge index."""
    row, col = edge_index
    key = (row << 32) | col
    rev_key = (col << 32) | row
    key_sorted, perm = key.sort()
    idx = torch.searchsorted(key_sorted, rev_key)
    return perm[idx]


def custom_scatter(src, index, dim_size, C, reduce):
    """Custom scatter function to conveniently aggregate features, falling back to scatter_reduce_."""
    out = src.new_zeros(dim_size, C)
    out.scatter_reduce_(
        0, index.unsqueeze(-1).expand(-1, C), src, reduce=reduce, include_self=False
    )
    return out
