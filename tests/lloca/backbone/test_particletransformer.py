import torch
import pytest
from tests.constants import (
    TOLERANCES,
    MILD_TOLERANCES,
    LOGM2_MEAN_STD,
    REPS,
    FRAMES_PREDICTOR,
)
from tests.helpers import sample_particle, equivectors_builder
from tests.hep import get_tagging_features

from lloca.backbone.particletransformer import ParticleTransformer, Block
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from lloca.framesnet.frames import InverseFrames
from lloca.framesnet.equi_frames import LearnedSO13Frames, LearnedPDFrames
from lloca.backbone.attention import LLoCaAttention


@pytest.mark.parametrize("FramesPredictor", FRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("attn_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_block_invariance_equivariance(
    FramesPredictor,
    batch_dims,
    logm2_std,
    logm2_mean,
    attn_reps,
    num_heads,
):
    dtype = torch.float64

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # define block
    in_reps = TensorReps("1x1n")
    attn_reps = TensorReps(attn_reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    linear_in = torch.nn.Linear(in_reps.dim, attn_reps.dim * num_heads).to(dtype=dtype)
    linear_out = torch.nn.Linear(attn_reps.dim * num_heads, in_reps.dim).to(dtype=dtype)
    attention = LLoCaAttention(attn_reps, num_heads)
    ParT_block = Block(attention=attention, embed_dim=attn_reps.dim * num_heads).to(
        dtype
    )
    ParT_block.eval()  # turn off dropout

    def block_wrapper(x, frames):
        x = x.unsqueeze(0)
        mask = torch.ones_like(x[..., 0])
        frames = frames.reshape(1, *frames.shape)
        attention.prepare_frames(frames)
        x = ParT_block(x=x, padding_mask=mask)
        x = x.squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    frames = call_predictor(fm)
    fm_local = trafo(fm, frames)

    # block - global
    x_local = linear_in(fm_local)
    x_prime_local = block_wrapper(x_local, frames)
    fm_prime_local = linear_out(x_prime_local)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # global - block
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    frames_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, frames_transformed)
    x_tr_local = linear_in(fm_tr_local)
    x_tr_prime_local = block_wrapper(x_tr_local, frames_transformed)
    fm_tr_prime_local = linear_out(x_tr_prime_local)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseFrames(frames_transformed))

    # test feature invariance before the operation
    torch.testing.assert_close(x_local, x_tr_local, **TOLERANCES)

    # test feature invariance after the operation
    torch.testing.assert_close(x_tr_prime_local, x_prime_local, **TOLERANCES)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)


@pytest.mark.parametrize(
    "FramesPredictor", [LearnedSO13Frames, LearnedPDFrames]
)  # RestFrames gives nans sometimes
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_ParT_invariance(
    FramesPredictor,
    batch_dims,
    logm2_std,
    logm2_mean,
):
    dtype = torch.float64
    batch = torch.zeros(batch_dims[0], dtype=torch.long)

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # define ParT
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    model = ParticleTransformer(
        input_dim=7,
        num_classes=1,
        attn_reps="8x0n+2x1n",
    ).to(dtype=dtype)
    model.eval()  # turn off dropout

    def ParT_wrapper(p_local, frames):
        fts_local = get_tagging_features(p_local, batch)
        fts_local = fts_local.transpose(-1, -2).unsqueeze(0)
        p_local = p_local[..., [1, 2, 3, 0]]
        p_local = p_local.transpose(-1, -2).unsqueeze(0)
        mask = torch.ones_like(p_local[..., [0], :])
        frames = frames.reshape(1, *frames.shape)
        x = model(x=fts_local, v=p_local, frames=frames, mask=mask)
        x = x.transpose(-1, -2).squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    frames = call_predictor(fm)
    fm_local = trafo(fm, frames)

    # ParT
    score_prime_local = ParT_wrapper(fm_local, frames)

    # global - ParT
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    frames_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, frames_transformed)
    score_tr_prime_local = ParT_wrapper(fm_tr_local, frames_transformed)

    # test feature invariance before the operation
    torch.testing.assert_close(fm_local, fm_tr_local, **TOLERANCES)

    # test equivariance of scores
    torch.testing.assert_close(
        score_tr_prime_local, score_prime_local, **MILD_TOLERANCES
    )
