import pytest
import math
import torch
from .utils import generate_batch, permute_single_graph

from pelican.nets import PELICAN


def run_shape_test(
    num_blocks,
    hidden_channels,
    increase_hidden_channels,
    in_channels_rank0,
    in_channels_rank1,
    in_channels_rank2,
    out_rank,
    out_channels,
    map_multipliers,
    factorize,
    compile,
    checkpoint_blocks,
):
    batch, edge_index, graph, _, _ = generate_batch(C=in_channels_rank0)
    _, _, _, nodes, _ = generate_batch(
        C=in_channels_rank1, batch=batch, edge_index=edge_index
    )
    _, _, _, _, edges = generate_batch(
        C=in_channels_rank2, batch=batch, edge_index=edge_index
    )
    G = batch[-1].item() + 1
    N = batch.size(0)
    E = edge_index.size(1)
    out_objs = {0: G, 1: N, 2: E}[out_rank]

    net = PELICAN(
        num_blocks=num_blocks,
        hidden_channels=hidden_channels,
        increase_hidden_channels=increase_hidden_channels,
        in_channels_rank0=in_channels_rank0,
        in_channels_rank1=in_channels_rank1,
        in_channels_rank2=in_channels_rank2,
        out_rank=out_rank,
        out_channels=out_channels,
        map_multipliers=map_multipliers,
        factorize=factorize,
        compile=compile,
        checkpoint_blocks=checkpoint_blocks,
    )
    out = net(
        in_rank2=edges,
        in_rank1=nodes,
        in_rank0=graph,
        batch=batch,
        edge_index=edge_index,
        num_graphs=G,
    )
    assert out.shape == (out_objs, out_channels)


@pytest.mark.parametrize(
    "hidden_channels,increase_hidden_channels", [(16, 1), (7, math.pi)]
)
@pytest.mark.parametrize("num_blocks", [0, 1, 3])
@pytest.mark.parametrize(
    "in_channels_rank0,in_channels_rank1,in_channels_rank2",
    [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)],
)
@pytest.mark.parametrize("out_rank,out_channels", [(0, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("map_multipliers", [True, False])
@pytest.mark.parametrize("factorize", [True, False])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_shape(
    num_blocks,
    hidden_channels,
    increase_hidden_channels,
    in_channels_rank0,
    in_channels_rank1,
    in_channels_rank2,
    out_rank,
    out_channels,
    map_multipliers,
    factorize,
    checkpoint_blocks,
    compile=False,
):
    run_shape_test(
        num_blocks,
        hidden_channels,
        increase_hidden_channels,
        in_channels_rank0,
        in_channels_rank1,
        in_channels_rank2,
        out_rank,
        out_channels,
        map_multipliers,
        factorize,
        compile,
        checkpoint_blocks,
    )


@pytest.mark.parametrize(
    "hidden_channels,increase_hidden_channels", [(16, 1), (7, math.pi)]
)
@pytest.mark.parametrize("num_blocks", [0, 1, 3])
@pytest.mark.parametrize(
    "in_channels_rank0,in_channels_rank1,in_channels_rank2",
    [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)],
)
@pytest.mark.parametrize("out_rank,out_channels", [(0, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("map_multipliers", [True, False])
@pytest.mark.parametrize("factorize", [True, False])
@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_permutation_equivariance(
    num_blocks,
    hidden_channels,
    increase_hidden_channels,
    in_channels_rank0,
    in_channels_rank1,
    in_channels_rank2,
    out_rank,
    out_channels,
    map_multipliers,
    factorize,
    checkpoint_blocks,
    compile=False,
):
    # only test permutation equivariance on single graphs for simplicity
    batch, edge_index, graph, _, _ = generate_batch(C=in_channels_rank0, G=1)
    _, _, _, nodes, _ = generate_batch(
        C=in_channels_rank1, batch=batch, edge_index=edge_index
    )
    _, _, _, _, edges = generate_batch(
        C=in_channels_rank2, batch=batch, edge_index=edge_index
    )
    G = batch[-1].item() + 1

    net = PELICAN(
        num_blocks=num_blocks,
        hidden_channels=hidden_channels,
        increase_hidden_channels=increase_hidden_channels,
        in_channels_rank0=in_channels_rank0,
        in_channels_rank1=in_channels_rank1,
        in_channels_rank2=in_channels_rank2,
        out_rank=out_rank,
        out_channels=out_channels,
        map_multipliers=map_multipliers,
        factorize=factorize,
        compile=compile,
        checkpoint_blocks=checkpoint_blocks,
    )

    # path 1: first aggregate, then permute
    out = net(
        in_rank2=edges,
        in_rank1=nodes,
        in_rank0=graph,
        batch=batch,
        edge_index=edge_index,
        num_graphs=G,
    )
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
        permutation=permutation,
        graph=graph,
        nodes=nodes,
        edges=edges,
        edge_index=edge_index,
    )
    out_permfirst = net(
        in_rank2=edges_perm,
        in_rank1=nodes_perm,
        in_rank0=graph_perm,
        batch=batch,
        edge_index=edge_index_perm,
        num_graphs=G,
    )

    torch.testing.assert_close(out_permlater, out_permfirst)


def test_compile(
    num_blocks=1,
    hidden_channels=16,
    increase_hidden_channels=1,
    in_channels_rank0=1,
    in_channels_rank1=1,
    in_channels_rank2=1,
    out_rank=2,
    out_channels=3,
    map_multipliers=True,
    factorize=False,
    checkpoint_blocks=False,
    compile=True,
):
    run_shape_test(
        num_blocks,
        hidden_channels,
        increase_hidden_channels,
        in_channels_rank0,
        in_channels_rank1,
        in_channels_rank2,
        out_rank,
        out_channels,
        map_multipliers,
        factorize,
        compile,
        checkpoint_blocks,
    )
