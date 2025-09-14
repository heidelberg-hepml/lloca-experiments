import torch
import pytest
from tests.constants import TOLERANCES, MILD_TOLERANCES, BATCH_DIMS
from tests.helpers import lorentz_test

from lloca.utils.rand_transforms import (
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
    rand_ztransform,
    rand_boost,
)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("std_eta", [0.1, 1, 2])
@pytest.mark.parametrize(
    "transform_type",
    [
        rand_lorentz,
        rand_rotation,
        rand_xyrotation,
        rand_ztransform,
        rand_boost,
    ],
)
def test_rand_lorentz(batch_dims, std_eta, transform_type):
    dtype = torch.float64

    # collect N different kinds of transformations
    kwargs = {
        "shape": batch_dims,
        "dtype": dtype,
    }
    if transform_type in [rand_lorentz]:
        kwargs["std_eta"] = std_eta
    transform = transform_type(**kwargs)

    # test that this is a valid Lorentz transform
    lorentz_test(transform, **MILD_TOLERANCES)

    # test specific properties of the transform
    if transform_type in [rand_rotation, rand_xyrotation]:
        should_zero = torch.cat(
            [transform[..., 0, 1:].flatten(), transform[..., 1:, 0].flatten()]
        )
        if transform_type == rand_xyrotation:
            should_zero = torch.cat(
                [
                    should_zero,
                    transform[..., 3, 1:2].flatten(),
                    transform[..., 1:2, 3].flatten(),
                ]
            )
        torch.testing.assert_close(
            should_zero, torch.zeros_like(should_zero), **TOLERANCES
        )
