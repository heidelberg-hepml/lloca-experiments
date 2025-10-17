import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS

from lloca.utils.lorentz import lorentz_inner, lorentz_squarednorm
from lloca.utils.orthogonalize_4d import orthogonalize_4d


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("method", ["cross", "gramschmidt"])
@pytest.mark.parametrize(
    "vector_type,eps",
    [
        ("naive", None),
        ("coplanar", 1e-2),
        ("coplanar", 1e-3),
        ("lightlike", 1e-2),
        ("lightlike", 1e-4),
    ],
)
def test_orthogonalize(batch_dims, method, vector_type, eps):
    # check orthogonality after using the function
    dtype = torch.float64

    v1 = torch.randn(batch_dims + [4], dtype=dtype)
    v2 = torch.randn(batch_dims + [4], dtype=dtype)
    if vector_type == "naive":
        # third vector is fully random
        v3 = torch.randn(batch_dims + [4], dtype=dtype)
    elif vector_type == "coplanar":
        # third vector is almost a linear combination of the first 2 vectors
        c1, c2 = torch.randn(batch_dims, dtype=dtype), torch.randn(
            batch_dims, dtype=dtype
        )
        v3 = (
            v1
            + c1.unsqueeze(-1) * v1
            + c2.unsqueeze(-1) * v2
            + eps * torch.randn_like(v1)
        )
    elif vector_type == "lightlike":
        # make all vectors lightlike
        v3s = torch.randn([3] + batch_dims + [3], dtype=dtype)
        norm = eps * torch.randn_like(v3s[..., [0]])
        v0 = torch.sqrt((v3s**2).sum(dim=-1, keepdim=True)) + norm
        vecs = torch.cat([v0, v3s], dim=-1)
        v1, v2, v3 = vecs

    vecs = torch.stack([v1, v2, v3], dim=-2)
    orthogonal_vecs = orthogonalize_4d(vecs, method=method)

    # test orthogonality
    n = orthogonal_vecs.ndim
    perm = [n - 2] + list(range(0, n - 2)) + [n - 1]
    orthogonal_vecs = orthogonal_vecs.permute(*perm)
    for i1, v1 in enumerate(orthogonal_vecs):
        for i2, v2 in enumerate(orthogonal_vecs):
            inner = lorentz_inner(v1, v2)
            target = torch.ones_like(inner) if i1 == i2 else torch.zeros_like(inner)
            torch.testing.assert_close(inner.abs(), target, **TOLERANCES)

    # check that there is only one time-like vector in each triplet of orthogonalized vectors
    norm = torch.stack([lorentz_squarednorm(v) for v in orthogonal_vecs], dim=-1)
    num_timelike = torch.sum(norm > 0, dim=-1)
    torch.testing.assert_close(
        num_timelike, torch.ones_like(num_timelike), atol=0, rtol=0
    )
