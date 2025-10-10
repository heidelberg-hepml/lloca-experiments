import pytest
import math
from .utils import generate_batch

from pelican.layers import (
    Aggregator2to2,
    Aggregator2to1,
    Aggregator2to0,
    Aggregator1to2,
    Aggregator0to2,
    PELICANBlock,
)


@pytest.mark.parametrize("aggr", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("map_multipliers", [True, False])
@pytest.mark.parametrize("factorize", [True, False])
@pytest.mark.parametrize(
    "aggregator,in_rank,out_rank",
    [
        [Aggregator0to2, 0, 2],
        [Aggregator1to2, 1, 2],
        [Aggregator2to0, 2, 0],
        [Aggregator2to1, 2, 1],
        [Aggregator2to2, 2, 2],
    ],
)
@pytest.mark.parametrize("in_channels,out_channels", [(16, 16), (7, 13)])
def test_shape_aggregator(
    aggregator,
    in_rank,
    out_rank,
    in_channels,
    out_channels,
    map_multipliers,
    factorize,
    aggr,
):
    batch, edge_index, graph, nodes, edges = generate_batch(C=in_channels)
    G = batch[-1].item() + 1
    N = batch.size(0)
    E = edge_index.size(1)

    in_data = {0: graph, 1: nodes, 2: edges}[in_rank]
    out_objs = {0: G, 1: N, 2: E}[out_rank]

    handler = aggregator(
        in_channels=in_channels,
        out_channels=out_channels,
        map_multipliers=map_multipliers,
        factorize=factorize,
        aggr=aggr,
    )
    out = handler(in_data, edge_index, batch, G=G)
    assert out.shape == (out_objs, out_channels)


@pytest.mark.parametrize("aggr", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize(
    "activation", ["relu", "gelu", "leaky_relu", "tanh", "sigmoid", "silu"]
)
@pytest.mark.parametrize("map_multipliers", [True, False])
@pytest.mark.parametrize("factorize", [True, False])
@pytest.mark.parametrize(
    "hidden_channels,increase_hidden_channels", [(16, 1), (13, math.pi), (19, 0.123)]
)
def test_shape_block(
    map_multipliers,
    factorize,
    aggr,
    activation,
    hidden_channels,
    increase_hidden_channels,
):
    batch, edge_index, _, _, edges = generate_batch(C=hidden_channels)
    E = edge_index.size(1)

    handler = PELICANBlock(
        hidden_channels=hidden_channels,
        increase_hidden_channels=increase_hidden_channels,
        activation=activation,
        map_multipliers=map_multipliers,
        factorize=factorize,
        aggr=aggr,
    )
    out = handler(edges, edge_index, batch)
    assert out.shape == (E, hidden_channels)
