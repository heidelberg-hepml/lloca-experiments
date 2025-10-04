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

from lloca.backbone.particlenet import EdgeConvBlock, ParticleNet
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from lloca.framesnet.frames import InverseFrames
from lloca.framesnet.equi_frames import LearnedSO13Frames, LearnedPDFrames


@pytest.mark.parametrize("FramesPredictor", FRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("k", [2, 5])
@pytest.mark.parametrize("out_feats", [(12, 12)])
@pytest.mark.parametrize("hidden_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_edgeconvblock_invariance_equivariance(
    FramesPredictor,
    batch_dims,
    k,
    out_feats,
    logm2_std,
    logm2_mean,
    hidden_reps,
):
    dtype = torch.float64

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # define edgeconv
    in_reps = TensorReps("1x1n")
    hidden_reps = TensorReps(hidden_reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    linear_in = torch.nn.Linear(in_reps.dim, hidden_reps.dim).to(dtype=dtype)
    linear_out = torch.nn.Linear(out_feats[-1], in_reps.dim).to(dtype=dtype)
    edgeconv = EdgeConvBlock(k=k, in_reps=hidden_reps, out_feats=out_feats).to(dtype)

    def edgeconvblock_wrapper(x, frames):
        # use features as points for simplicity
        # this is equivariant, because features in local frame are invariant
        # and hence the knn ordering on them is also invariant
        # have to reshape to match ParticleNet format
        x = x.transpose(-1, -2).unsqueeze(0)
        x = edgeconv(points=x, features=x, frames=frames)
        x = x.transpose(-1, -2).squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    frames = call_predictor(fm)
    fm_local = trafo(fm, frames)

    # edgeconv - global
    x_local = linear_in(fm_local)
    x_prime_local = edgeconvblock_wrapper(x_local, frames)
    fm_prime_local = linear_out(x_prime_local)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # global - edgeconv
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    frames_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, frames_transformed)
    x_tr_local = linear_in(fm_tr_local)
    x_tr_prime_local = edgeconvblock_wrapper(x_tr_local, frames_transformed)
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
def test_particlenet_invariance(
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

    # define particlenet
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    hidden_reps_list = ["3x0n+1x1n", "16x0n+4x1n"]  # pick something
    particlenet = ParticleNet(
        input_dims=TensorReps(hidden_reps_list[0]).dim,
        hidden_reps_list=hidden_reps_list,
        num_classes=1,
    ).to(dtype=dtype)
    particlenet.eval()  # turn off dropout

    def edgeconvblock_wrapper(p_local, frames):
        fts_local = get_tagging_features(p_local, batch)
        fts_local = fts_local.transpose(-1, -2).unsqueeze(0)
        points_local = fts_local[:, [4, 5], :]
        x = particlenet(points=points_local, features=fts_local, frames=frames)
        x = x.transpose(-1, -2).squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    frames = call_predictor(fm)
    fm_local = trafo(fm, frames)

    # particlenet
    score_prime_local = edgeconvblock_wrapper(fm_local, frames)

    # global - particlenet
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    frames_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, frames_transformed)
    score_tr_prime_local = edgeconvblock_wrapper(fm_tr_local, frames_transformed)

    # test feature invariance before the operation
    torch.testing.assert_close(fm_local, fm_tr_local, **TOLERANCES)

    # test equivariance of scores
    torch.testing.assert_close(
        score_tr_prime_local, score_prime_local, **MILD_TOLERANCES
    )
