import torch


def bell_number(n):
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
    _, C = edges.shape
    N = batch.size(0)
    row, col = edge_index
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags = edges * diag_mask
    row_agg = custom_scatter(edges, row, dim_size=N, C=C, reduce=reduce)
    col_agg = custom_scatter(edges, col, dim_size=N, C=C, reduce=reduce)
    graph_agg = custom_scatter(edges, edge_batch, dim_size=N, C=C, reduce=reduce)
    diag_agg = custom_scatter(diags, row, dim_size=N, C=C, reduce=reduce)

    ops = torch.stack(
        [
            edges[is_diag],
            row_agg,
            col_agg,
            graph_agg[batch],
            diag_agg[batch],
        ],
        dim=-1,
    )  # shape (N, C, 5)
    return ops


def aggregate_2to2(edges, edge_index, batch, reduce="mean", **kwargs):
    _, C = edges.shape
    N = batch.size(0)
    row, col = edge_index
    perm_T = get_transpose(edge_index)
    edge_batch = batch[row]
    is_diag = row == col
    diag_mask = is_diag.unsqueeze(-1).type_as(edges)

    diags = edges * diag_mask

    row_agg = custom_scatter(edges, row, dim_size=N, C=C, reduce=reduce)
    col_agg = custom_scatter(edges, col, dim_size=N, C=C, reduce=reduce)
    graph_agg = custom_scatter(edges, edge_batch, dim_size=N, C=C, reduce=reduce)
    diag_agg = custom_scatter(diags, row, dim_size=N, C=C, reduce=reduce)

    # Note: creating ops as empty tensor and filling it is ~4% faster but requires ~10% more RAM
    ops = torch.stack(
        [
            edges,
            edges[perm_T],
            diags,
            diag_agg[row],
            diag_agg[col],
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
    row, col = edge_index
    key = (row << 32) | col
    rev_key = (col << 32) | row
    key_sorted, perm = key.sort()
    idx = torch.searchsorted(key_sorted, rev_key)
    return perm[idx]


def custom_scatter(src, index, dim_size, C, reduce):
    out = src.new_zeros(dim_size, C)
    out.scatter_reduce_(
        0, index.unsqueeze(-1).expand(-1, C), src, reduce=reduce, include_self=False
    )
    return out
