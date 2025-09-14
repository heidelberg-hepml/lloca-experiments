import torch
import pytest
from tests.constants import TOLERANCES, REPS

from lloca.framesnet.frames import Frames, InverseFrames, ChangeOfFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz


@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("reps", REPS)
def test_equivariance(batch_dims, reps):
    dtype = torch.float64

    reps = TensorReps(reps)
    trafo = TensorRepsTransform(TensorReps(reps))

    transform = rand_lorentz(batch_dims, dtype=dtype)
    frames = Frames(transform)

    x = torch.randn(*batch_dims, reps.dim, dtype=dtype)

    # manual transform
    transform_direct = torch.einsum(
        "...ij,...jk->...ik", frames.matrices, InverseFrames(frames).matrices
    )
    change_frames1 = Frames(transform_direct)
    x_prime1 = trafo(x, change_frames1)
    torch.testing.assert_close(x, x_prime1, **TOLERANCES)

    # all-in-one transform
    change_frames2 = ChangeOfFrames(frames, frames)
    x_prime2 = trafo(x, change_frames2)
    torch.testing.assert_close(x, x_prime2, **TOLERANCES)
