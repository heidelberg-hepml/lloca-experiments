import pytest
import math
from .utils import generate_batch

from pelican.nets import PELICAN


def run_test(
    num_blocks,
    hidden_channels,
    increase_hidden_channels,
    in_channels_rank0,
    in_channels_rank1,
    in_channels_rank2,
    out_rank,
    out_channels,
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

    if out_rank == 0:
        out_objs = G
    elif out_rank == 1:
        out_objs = N
    elif out_rank == 2:
        out_objs = E
    else:
        raise ValueError(f"Unsupported out_rank={out_rank}")

    handler = PELICAN(
        num_blocks=num_blocks,
        hidden_channels=hidden_channels,
        increase_hidden_channels=increase_hidden_channels,
        in_channels_rank0=in_channels_rank0,
        in_channels_rank1=in_channels_rank1,
        in_channels_rank2=in_channels_rank2,
        out_rank=out_rank,
        out_channels=out_channels,
        compile=compile,
        checkpoint_blocks=checkpoint_blocks,
    )
    out = handler(
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
    checkpoint_blocks,
    compile=False,
):
    run_test(
        num_blocks,
        hidden_channels,
        increase_hidden_channels,
        in_channels_rank0,
        in_channels_rank1,
        in_channels_rank2,
        out_rank,
        out_channels,
        compile,
        checkpoint_blocks,
    )


def test_compile(
    num_blocks=1,
    hidden_channels=16,
    increase_hidden_channels=1,
    in_channels_rank0=1,
    in_channels_rank1=1,
    in_channels_rank2=1,
    out_rank=2,
    out_channels=3,
    checkpoint_blocks=False,
    compile=True,
):
    run_test(
        num_blocks,
        hidden_channels,
        increase_hidden_channels,
        in_channels_rank0,
        in_channels_rank1,
        in_channels_rank2,
        out_rank,
        out_channels,
        compile,
        checkpoint_blocks,
    )
