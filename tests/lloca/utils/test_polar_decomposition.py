import torch
import pytest
from tests.constants import TOLERANCES, BATCH_DIMS, LOGM2_MEAN_STD
from tests.helpers import sample_particle, lorentz_test

from lloca.utils.polar_decomposition import (
    restframe_boost,
    polar_decomposition,
)
from lloca.utils.lorentz import lorentz_squarednorm
from lloca.utils.rand_transforms import rand_lorentz, rand_rotation


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "restframe_transform",
    [restframe_boost, polar_decomposition],
)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_restframe(batch_dims, restframe_transform, logm2_std, logm2_mean):
    dtype = torch.float64

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    kwargs = {}
    if restframe_transform == polar_decomposition:
        kwargs["references"] = [
            sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
            for _ in range(2)
        ]
        kwargs["return_reg"] = False

    # determine transformation into rest frame
    rest_trafo = restframe_transform(fm, **kwargs)
    fm_rest = torch.einsum("...ij,...j->...i", rest_trafo, fm)

    # check that the transformed fourmomenta are in the rest frame,
    # i.e. their spatial components vanish and the temporal component is the mass
    torch.testing.assert_close(
        fm_rest[..., 1:], torch.zeros_like(fm[..., 1:]), **TOLERANCES
    )
    torch.testing.assert_close(
        fm_rest[..., 0] ** 2, lorentz_squarednorm(fm), **TOLERANCES
    )

    lorentz_test(rest_trafo, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("random_transform", [rand_lorentz, rand_rotation])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_restframe_transformation(batch_dims, random_transform, logm2_std, logm2_mean):
    dtype = torch.float64

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    references = [
        sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
        for _ in range(2)
    ]

    # determine transformation into rest frame
    rest_trafo = polar_decomposition(fm, references, return_reg=False)
    fm_rest = torch.einsum("...ij,...j->...i", rest_trafo, fm)

    # random global transformation
    random = random_transform([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    references_prime = [torch.einsum("...ij,...j->...i", random, r) for r in references]
    rest_trafo_prime = polar_decomposition(fm_prime, references_prime, return_reg=False)
    fm_rest_prime = torch.einsum("...ij,...j->...i", rest_trafo_prime, fm_prime)

    # check that fourmomenta in rest frame is invariant
    torch.testing.assert_close(fm_rest, fm_rest_prime, **TOLERANCES)

    # check that the restframe transformations transform as expected
    # expect rest_trafo_prime = rest_trafo * random^-1
    # or rest_trafo_prime * random = rest_trafo
    rest_trafo_expected = torch.einsum("...ij,...jk->...ik", rest_trafo_prime, random)

    torch.testing.assert_close(rest_trafo, rest_trafo_expected, **TOLERANCES)
