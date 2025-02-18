import torch
import pytest
from tests.constants import (
    TOLERANCES,
    MILD_TOLERANCES,
    LOGM2_MEAN_STD,
    REPS,
    LFRAMES_PREDICTOR,
)
from tests.helpers import sample_particle
from torch_geometric.utils import dense_to_sparse

from tensorframes.nn.particlenet import EdgeConvBlock, TFParticleNet
from tensorframes.reps.tensorreps import TensorReps
from tensorframes.reps.tensorreps_transform import TensorRepsTransform
from tensorframes.utils.transforms import rand_lorentz
from tensorframes.lframes.lframes import InverseLFrames
from tensorframes.lframes.equi_lframes import (
    OrthogonalLearnedLFrames,
    LearnedRestLFrames,
)

from experiments.tagging.embedding import get_tagging_features


@pytest.mark.parametrize("LFramesPredictor", LFRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("k", [2, 5])
@pytest.mark.parametrize("out_feats", [(12, 12)])
@pytest.mark.parametrize("hidden_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_edgeconvblock_invariance_equivariance(
    LFramesPredictor,
    batch_dims,
    k,
    out_feats,
    logm2_std,
    logm2_mean,
    hidden_reps,
):
    dtype = torch.float64

    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]

    assert len(batch_dims) == 1
    predictor = LFramesPredictor(hidden_channels=16, num_layers=1, in_nodes=0).to(
        dtype=dtype
    )
    batch = torch.zeros(batch_dims, dtype=torch.long)
    scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
    call_predictor = lambda fm: predictor(fm, scalars, edge_index, batch)

    # define edgeconv
    in_reps = TensorReps("1x1n")
    hidden_reps = TensorReps(hidden_reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    linear_in = torch.nn.Linear(in_reps.dim, hidden_reps.dim).to(dtype=dtype)
    linear_out = torch.nn.Linear(out_feats[-1], in_reps.dim).to(dtype=dtype)
    edgeconv = EdgeConvBlock(k=k, in_reps=hidden_reps, out_feats=out_feats).to(dtype)

    def edgeconvblock_wrapper(x, lframes):
        # use features as points for simplicity
        # this is equivariant, because features in local frame are invariant
        # and hence the knn ordering on them is also invariant
        # have to reshape to match ParticleNet format
        x = x.transpose(-1, -2).unsqueeze(0)
        x = edgeconv(points=x, features=x, lframes=lframes)
        x = x.transpose(-1, -2).squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    lframes = call_predictor(fm)
    fm_local = trafo(fm, lframes)

    # edgeconv - global
    x_local = linear_in(fm_local)
    x_prime_local = edgeconvblock_wrapper(x_local, lframes)
    fm_prime_local = linear_out(x_prime_local)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseLFrames(lframes))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # global - edgeconv
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    lframes_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, lframes_transformed)
    x_tr_local = linear_in(fm_tr_local)
    x_tr_prime_local = edgeconvblock_wrapper(x_tr_local, lframes_transformed)
    fm_tr_prime_local = linear_out(x_tr_prime_local)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseLFrames(lframes_transformed))

    # test feature invariance before the operation
    torch.testing.assert_close(x_local, x_tr_local, **TOLERANCES)

    # test feature invariance after the operation
    torch.testing.assert_close(x_tr_prime_local, x_prime_local, **TOLERANCES)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)


@pytest.mark.parametrize(
    "LFramesPredictor", [OrthogonalLearnedLFrames, LearnedRestLFrames]
)  # RestLFrames gives nans sometimes
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_particlenet_invariance(
    LFramesPredictor,
    batch_dims,
    logm2_std,
    logm2_mean,
):
    dtype = torch.float64

    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]

    assert len(batch_dims) == 1
    predictor = LFramesPredictor(hidden_channels=16, num_layers=1, in_nodes=0).to(
        dtype=dtype
    )
    batch = torch.zeros(batch_dims, dtype=torch.long)
    scalars = torch.zeros(*batch_dims, 0, dtype=dtype)
    call_predictor = lambda fm: predictor(fm, scalars, edge_index, batch)

    # define particlenet
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    hidden_reps_list = ["3x0n+1x1n", "16x0n+4x1n"]  # pick something
    particlenet = TFParticleNet(
        hidden_reps_list=hidden_reps_list,
        num_classes=1,
    ).to(dtype=dtype)
    particlenet.eval()  # turn off dropout

    def edgeconvblock_wrapper(p_local, lframes):
        fts_local = get_tagging_features(p_local, batch)
        fts_local = fts_local.transpose(-1, -2).unsqueeze(0)
        points_local = fts_local[:, [4, 5], :]
        x = particlenet(points=points_local, features=fts_local, lframes=lframes)
        x = x.transpose(-1, -2).squeeze(0)
        return x

    # get global transformation
    random = rand_lorentz([1], dtype=dtype)
    random = random.repeat(*batch_dims, 1, 1)

    # sample Lorentz vectors
    fm = sample_particle(batch_dims, logm2_std, logm2_mean, dtype=dtype)
    lframes = call_predictor(fm)
    fm_local = trafo(fm, lframes)

    # particlenet
    score_prime_local = edgeconvblock_wrapper(fm_local, lframes)

    # global - particlenet
    fm_transformed = torch.einsum("...ij,...j->...i", random, fm)
    lframes_transformed = call_predictor(fm_transformed)
    fm_tr_local = trafo(fm_transformed, lframes_transformed)
    score_tr_prime_local = edgeconvblock_wrapper(fm_tr_local, lframes_transformed)

    # test feature invariance before the operation
    torch.testing.assert_close(fm_local, fm_tr_local, **TOLERANCES)

    # test equivariance of scores
    torch.testing.assert_close(
        score_tr_prime_local, score_prime_local, **MILD_TOLERANCES
    )
