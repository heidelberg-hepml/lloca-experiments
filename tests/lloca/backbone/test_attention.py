import torch
import pytest
from torch.nn import Linear
from tests.constants import TOLERANCES, LOGM2_MEAN_STD, REPS, FRAMES_PREDICTOR
from tests.helpers import sample_particle, equivectors_builder

from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.backbone.attention import LLoCaAttention
from lloca.framesnet.frames import InverseFrames
from lloca.utils.rand_transforms import rand_lorentz


@pytest.mark.parametrize("FramesPredictor", FRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("hidden_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_invariance_equivariance(
    FramesPredictor,
    batch_dims,
    hidden_reps,
    logm2_std,
    logm2_mean,
):
    dtype = torch.float64

    # preparations
    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # preparations
    in_reps = TensorReps("1x1n")
    hidden_reps = TensorReps(hidden_reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    attention = LLoCaAttention(hidden_reps, 1).to(dtype=dtype)
    linear_in = Linear(in_reps.dim, 3 * hidden_reps.dim).to(dtype=dtype)
    linear_out = Linear(hidden_reps.dim, in_reps.dim).to(dtype=dtype)

    # random global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)

    # path 1: Frames transform + random transform
    frames = call_predictor(fm)
    attention.prepare_frames(frames)
    fm_local = trafo(fm, frames)
    x_local = linear_in(fm_local).unsqueeze(0)
    q_local, k_local, v_local = x_local.chunk(3, dim=-1)
    x_local2 = attention(q_local, k_local, v_local).squeeze(0)
    fm_local = linear_out(x_local2)
    fm_global = trafo(fm_local, InverseFrames(frames))
    fm_global_prime = torch.einsum("...ij,...j->...i", random, fm_global)

    # path 2: random transform + Frames transform
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    frames_prime = call_predictor(fm_prime)
    attention.prepare_frames(frames_prime)
    fm_prime_local = trafo(fm_prime, frames_prime)
    x_prime_local = linear_in(fm_prime_local).unsqueeze(0)
    q_prime_local, k_prime_local, v_prime_local = x_prime_local.chunk(3, dim=-1)
    x_prime_local2 = attention(q_prime_local, k_prime_local, v_prime_local).squeeze(0)
    fm_prime_local = linear_out(x_prime_local2)
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames_prime))

    # test feature invariance before the operation
    torch.testing.assert_close(x_local, x_prime_local, **TOLERANCES)

    # test feature invariance after the operation
    torch.testing.assert_close(x_local2, x_prime_local2, **TOLERANCES)

    # test equivariance of output
    torch.testing.assert_close(fm_prime_global, fm_global_prime, **TOLERANCES)
