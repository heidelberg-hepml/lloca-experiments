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

    if in_rank == 0:
        in_data = graph
    elif in_rank == 1:
        in_data = nodes
    elif in_rank == 2:
        in_data = edges
    else:
        raise ValueError(f"Unsupported in_rank={in_rank}")

    if out_rank == 0:
        out_objs = G
    elif out_rank == 1:
        out_objs = N
    elif out_rank == 2:
        out_objs = E
    else:
        raise ValueError(f"Unsupported out_rank={out_rank}")

    out = aggregator(in_data, edge_index, batch, reduce=reduce, G=G)
    assert out.shape[:2] == (out_objs, C)
