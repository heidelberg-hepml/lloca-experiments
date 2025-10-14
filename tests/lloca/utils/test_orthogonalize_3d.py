import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS

from lloca.utils.orthogonalize_3d import orthogonalize_3d


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("method", ["cross", "gramschmidt"])
@pytest.mark.parametrize(
    "vector_type,eps",
    [("naive", None), ("collinear", 1e-2), ("collinear", 1e-3), ("collinear", 1e-4)],
)
def test_orthogonalize_o3(batch_dims, method, vector_type, eps):
    dtype = torch.float64

    v1 = torch.randn(batch_dims + [3], dtype=dtype)
    if vector_type == "naive":
        v2 = torch.randn_like(v1)
    elif vector_type == "collinear":
        v2 = v1 + eps * torch.randn_like(v1)
    else:
        raise ValueError(f"vector_type {vector_type} not implemented")

    vecs = torch.stack([v1, v2], dim=-2)
    orthogonal_vecs = orthogonalize_3d(vecs, method=method)
    orthogonal_vecs = orthogonal_vecs.unbind(dim=-2)

    # test orthonormality
    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = (v1 * v2).sum(dim=-1)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner, target, **TOLERANCES)
