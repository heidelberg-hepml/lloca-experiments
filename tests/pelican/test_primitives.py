import pytest
import torch
from .utils import generate_batch, permute_single_graph

from pelican.primitives import (
    aggregate_0to2,
    aggregate_1to2,
    aggregate_2to0,
    aggregate_2to1,
    aggregate_2to2,
)


@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize(
    "aggregator,in_rank,out_rank",
    [
        [aggregate_0to2, 0, 2],
        [aggregate_1to2, 1, 2],
        [aggregate_2to0, 2, 0],
        [aggregate_2to1, 2, 1],
        [aggregate_2to2, 2, 2],
    ],
)
def test_shape(aggregator, in_rank, out_rank, reduce):
    batch, edge_index, graph, nodes, edges = generate_batch()
    G = batch[-1].item() + 1
    N = batch.size(0)
    E = edge_index.size(1)
    C = graph.size(1)

    in_data = {0: graph, 1: nodes, 2: edges}[in_rank]
    out_objs = {0: G, 1: N, 2: E}[out_rank]

    out = aggregator(in_data, edge_index, batch, reduce=reduce, G=G)
    assert out.shape[:2] == (out_objs, C)


@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize(
    "aggregator,in_rank,out_rank",
    [
        [aggregate_0to2, 0, 2],
        [aggregate_1to2, 1, 2],
        [aggregate_2to0, 2, 0],
        [aggregate_2to1, 2, 1],
        [aggregate_2to2, 2, 2],
    ],
)
def test_permutation_equivariance(aggregator, in_rank, out_rank, reduce):
    # only test permutation equivariance on single graphs for simplicity
    batch, edge_index, graph, nodes, edges = generate_batch(G=1, N_range=[2, 3], C=1)
    G = batch[-1].item() + 1

    # path 1: first aggregate, then permute
    in_data = {0: graph, 1: nodes, 2: edges}[in_rank]
    out = aggregator(in_data, edge_index, batch, reduce=reduce, G=G)
    kwargs = {
        0: {"graph": out, "nodes": None, "edges": None},
        1: {"graph": None, "nodes": out, "edges": None},
        2: {"graph": None, "nodes": None, "edges": out},
    }[out_rank]
    (
        permutation,
        graph_perm,
        nodes_perm,
        edges_perm,
        edge_index_perm,
    ) = permute_single_graph(**kwargs, edge_index=edge_index)
    out_permlater = {0: graph_perm, 1: nodes_perm, 2: edges_perm}[out_rank]

    # path 2: first permute, then aggregate
    _, graph_perm, nodes_perm, edges_perm, _ = permute_single_graph(
        graph=graph,
        nodes=nodes,
        edges=edges,
        edge_index=edge_index,
        permutation=permutation,
    )
    in_data_perm = {0: graph_perm, 1: nodes_perm, 2: edges_perm}[in_rank]
    out_permfirst = aggregator(in_data_perm, edge_index_perm, batch, reduce=reduce, G=G)

    torch.testing.assert_close(out_permlater, out_permfirst)
