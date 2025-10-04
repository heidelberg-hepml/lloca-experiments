import torch
import pytest

from lloca.utils.utils import get_edge_index_from_ptr, get_edge_index_from_shape


@pytest.mark.parametrize("B, N", [(1, 5), (4, 9)])
def test_get_edge_index_tools(B, N):
    # test that the two get_edge_index functions give the same result
    tensor = torch.randn(B, N, 16)
    ptr = torch.arange(B + 1) * N

    edge_index_from_shape, _ = get_edge_index_from_shape(tensor)
    edge_index_from_ptr = get_edge_index_from_ptr(ptr)

    assert torch.all(edge_index_from_shape == edge_index_from_ptr)
