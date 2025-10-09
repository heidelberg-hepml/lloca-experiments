import pytest
from .utils import generate_batch

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
