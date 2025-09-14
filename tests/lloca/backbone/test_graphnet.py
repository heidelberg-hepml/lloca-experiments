import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN_STD, REPS, FRAMES_PREDICTOR
from tests.helpers import sample_particle, equivectors_builder
from torch_geometric.utils import dense_to_sparse

from lloca.backbone.graphnet import EdgeConv, GraphNet
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.rand_transforms import rand_lorentz
from lloca.framesnet.frames import InverseFrames


@pytest.mark.parametrize("FramesPredictor", FRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_layers_mlp1", range(1, 2))
@pytest.mark.parametrize("num_layers_mlp2", range(0, 2))
@pytest.mark.parametrize("hidden_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_edgeconv_invariance_equivariance(
    FramesPredictor,
    batch_dims,
    num_layers_mlp1,
    num_layers_mlp2,
    logm2_std,
    logm2_mean,
    hidden_reps,
):
    dtype = torch.float64

    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # define edgeconv
    in_reps = TensorReps("1x1n")
    hidden_reps = TensorReps(hidden_reps)
    trafo = TensorRepsTransform(TensorReps(in_reps))
    linear_in = torch.nn.Linear(in_reps.dim, hidden_reps.dim).to(dtype=dtype)
    linear_out = torch.nn.Linear(hidden_reps.dim, in_reps.dim).to(dtype=dtype)
    edgeconv = EdgeConv(hidden_reps, num_layers_mlp1, num_layers_mlp2).to(dtype)

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
    x_tr_local = linear_in(fm_tr_local)
    x_tr_prime_local = edgeconv(x_tr_local, frames_transformed, edge_index)
    fm_tr_prime_local = linear_out(x_tr_prime_local)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseFrames(frames_transformed))

    # edgeconv - global
    x_local = linear_in(fm_local)
    x_prime_local = edgeconv(x_local, frames, edge_index)
    fm_prime_local = linear_out(x_prime_local)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # test feature invariance before the operation
    torch.testing.assert_close(x_local, x_tr_local, **TOLERANCES)

    # test feature invariance after the operation
    torch.testing.assert_close(x_tr_prime_local, x_prime_local, **TOLERANCES)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)


@pytest.mark.parametrize("FramesPredictor", FRAMES_PREDICTOR)
@pytest.mark.parametrize("batch_dims", [[10]])
@pytest.mark.parametrize("num_layers_mlp1", range(1, 2))
@pytest.mark.parametrize("num_layers_mlp2", range(0, 2))
@pytest.mark.parametrize("num_blocks", [0, 1, 2])
@pytest.mark.parametrize("hidden_reps", REPS)
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
def test_graphnet_invariance_equivariance(
    FramesPredictor,
    batch_dims,
    num_layers_mlp1,
    num_layers_mlp2,
    num_blocks,
    logm2_std,
    logm2_mean,
    hidden_reps,
):
    dtype = torch.float64

    edge_index = dense_to_sparse(torch.ones(batch_dims[0], batch_dims[0]))[0]

    assert len(batch_dims) == 1
    equivectors = equivectors_builder()
    predictor = FramesPredictor(equivectors=equivectors).to(dtype=dtype)
    call_predictor = lambda fm: predictor(fm)

    # define edgeconv
    in_reps = TensorReps("1x1n")
    trafo = TensorRepsTransform(TensorReps(in_reps))
    graphnet = GraphNet(
        in_channels=in_reps.dim,
        hidden_reps=hidden_reps,
        out_channels=in_reps.dim,
        num_blocks=num_blocks,
        num_layers_mlp1=num_layers_mlp1,
        num_layers_mlp2=num_layers_mlp2,
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
    fm_tr_prime_local = graphnet(fm_tr_local, frames_transformed, edge_index)
    # back to global frame
    fm_tr_prime_global = trafo(fm_tr_prime_local, InverseFrames(frames_transformed))

    # edgeconv - global
    fm_prime_local = graphnet(fm_local, frames, edge_index)
    # back to global
    fm_prime_global = trafo(fm_prime_local, InverseFrames(frames))
    fm_prime_tr_global = torch.einsum("...ij,...j->...i", random, fm_prime_global)

    # test equivariance of outputs
    torch.testing.assert_close(fm_tr_prime_global, fm_prime_tr_global, **TOLERANCES)
