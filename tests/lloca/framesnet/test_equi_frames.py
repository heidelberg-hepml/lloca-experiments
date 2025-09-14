import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN_STD
from tests.helpers import sample_particle, lorentz_test, equivectors_builder

from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz, rand_rotation, rand_xyrotation
from lloca.framesnet.frames import Frames
from lloca.framesnet.equi_frames import (
    LearnedSO13Frames,
    LearnedRestFrames,
    LearnedPDFrames,
    LearnedSO3Frames,
    LearnedSO2Frames,
)


@pytest.mark.parametrize(
    "FramesPredictor,rand_trafo",
    [
        (LearnedSO13Frames, rand_lorentz),
        (LearnedRestFrames, rand_lorentz),
        (LearnedPDFrames, rand_lorentz),
        (LearnedSO3Frames, rand_rotation),
        (LearnedSO2Frames, rand_xyrotation),
    ],
)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_frames_transformation(
    FramesPredictor, rand_trafo, batch_dims, logm2_std, logm2_mean
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # frames for un-transformed fm
    frames = call_predictor(fm)
    lorentz_test(frames.matrices, **TOLERANCES)

    # random global transformation
    random = rand_trafo([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # frames for transformed fm
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    frames_prime = call_predictor(fm_prime)
    lorentz_test(frames_prime.matrices, **TOLERANCES)

    # check that frames transform correctly
    # expect frames_prime = frames * random^-1
    inv_random = Frames(random).inv
    frames_prime_expected = torch.einsum(
        "...ij,...jk->...ik", frames.matrices, inv_random
    )
    torch.testing.assert_close(
        frames_prime_expected, frames_prime.matrices, **TOLERANCES
    )


@pytest.mark.parametrize(
    "FramesPredictor,rand_trafo",
    [
        (LearnedSO13Frames, rand_lorentz),
        (LearnedRestFrames, rand_lorentz),
        (LearnedPDFrames, rand_lorentz),
        (LearnedSO3Frames, rand_rotation),
        (LearnedSO2Frames, rand_xyrotation),
    ],
)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_feature_invariance(
    FramesPredictor, rand_trafo, batch_dims, logm2_std, logm2_mean
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(reps))

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # random global transformation
    random = rand_trafo([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # path 1: Frames transform (+ random transform)
    frames = call_predictor(fm)
    lorentz_test(frames.matrices, **TOLERANCES)
    fm_local = trafo(fm, frames)

    # path 2: random transform + Frames transform
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    frames_prime = call_predictor(fm_prime)
    lorentz_test(frames_prime.matrices, **TOLERANCES)
    fm_local_prime = trafo(fm_prime, frames_prime)

    # test that features are invariant
    torch.testing.assert_close(fm_local, fm_local_prime, **TOLERANCES)
