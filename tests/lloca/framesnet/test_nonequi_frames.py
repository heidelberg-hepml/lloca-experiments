import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN_STD
from tests.helpers import sample_particle

from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.framesnet.nonequi_frames import (
    IdentityFrames,
    RandomFrames,
)


@pytest.mark.parametrize(
    "FramesPredictor,transform_type",
    [
        (IdentityFrames, None),
        (RandomFrames, "lorentz"),
        (RandomFrames, "rotation"),
        (RandomFrames, "xyrotation"),
    ],
)
@pytest.mark.parametrize("batch_dims", [[1000]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_vectors(FramesPredictor, transform_type, batch_dims, logm2_mean, logm2_std):
    dtype = torch.float32

    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # predict local frames
    predictor = (
        FramesPredictor(transform_type=transform_type)
        if FramesPredictor == RandomFrames
        else FramesPredictor()
    )
    frames = predictor(fm)

    # transform into local frames
    reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(reps))
    fm_local = trafo(fm, frames)

    if FramesPredictor == IdentityFrames:
        # fourmomenta should not change
        torch.testing.assert_close(fm_local, fm, **TOLERANCES)
    elif type == "rotation":
        # energy and pz should not change
        torch.testing.assert_close(fm_local[..., [0]], fm[..., [0]], **TOLERANCES)
    elif type == "xyrotation":
        # energy and pz should not change
        torch.testing.assert_close(fm_local[..., [0, 3]], fm[..., [0, 3]], **TOLERANCES)
