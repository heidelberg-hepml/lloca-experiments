import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN_STD, REPS, FRAMES_PREDICTOR
from tests.helpers import sample_particle, equivectors_builder

from lloca.backbone.transformer import Transformer
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from lloca.framesnet.frames import InverseFrames


@pytest.mark.parametrize("FramesPredictor", FRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("num_blocks", [0, 1, 2])
@pytest.mark.parametrize("attn_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_transformer_invariance_equivariance(
    FramesPredictor,
    batch_dims,
    num_heads,
    num_blocks,
    logm2_std,
    logm2_mean,
    attn_reps,
):
    dtype = torch.float64

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # define edgeconv
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    net = Transformer(
        in_channels=in_reps.dim,
        attn_reps=attn_reps,
        out_channels=in_reps.dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    ).to(dtype=dtype)

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    frames = call_predictor(fm)
    fm_local = trafo(fm, frames)

    # global - edgeconv
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    frames_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, frames_transformed)
    fm_tr_prime_local = net(fm_tr_local, frames_transformed)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseFrames(frames_transformed))

    # edgeconv - global
    fm_prime_local = net(fm_local, frames)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)
