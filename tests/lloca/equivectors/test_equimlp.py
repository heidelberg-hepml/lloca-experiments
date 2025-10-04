import torch
import pytest
from tests.constants import TOLERANCES, LOGM2_MEAN_STD
from tests.helpers import sample_particle

from lloca.equivectors.equimlp import EquiMLP
from lloca.utils.rand_transforms import rand_lorentz


@pytest.mark.parametrize("batch_dims", [[100]])
@pytest.mark.parametrize("jet_size", [10])
@pytest.mark.parametrize("n_vectors", [1, 2, 3])
@pytest.mark.parametrize("hidden_channels,num_layers_mlp", [(16, 1)])
@pytest.mark.parametrize("logm2_mean,logm2_std", LOGM2_MEAN_STD)
@pytest.mark.parametrize("num_scalars,include_edges", [(0, True), (1, False)])
@pytest.mark.parametrize(
    "operation, fm_norm",
    [("diff", True), ("diff", False), ("add", True), ("add", False), ("single", False)],
)
@pytest.mark.parametrize("nonlinearity", ["softplus", "exp", "softmax", "softmax_safe"])
def test_equivariance(
    batch_dims,
    jet_size,
    n_vectors,
    hidden_channels,
    num_layers_mlp,
    logm2_std,
    logm2_mean,
    include_edges,
    num_scalars,
    operation,
    nonlinearity,
    fm_norm,
):
    assert len(batch_dims) == 1
    dtype = torch.float64

    # construct sparse tensors containing a set of equal-multiplicity jets
    ptr = torch.arange(0, (batch_dims[0] + 1) * jet_size, jet_size)

    # input to mlp: only edge attributes
    calc_node_attr = lambda fm: torch.zeros(*fm.shape[:-1], num_scalars, dtype=dtype)
    equivectors = EquiMLP(
        n_vectors=n_vectors,
        num_scalars=num_scalars,
        hidden_channels=hidden_channels,
        num_layers_mlp=num_layers_mlp,
        include_edges=include_edges,
        operation=operation,
        nonlinearity=nonlinearity,
        fm_norm=fm_norm,
    ).to(dtype=dtype)

    fm = sample_particle(
        batch_dims + [jet_size], logm2_std, logm2_mean, dtype=dtype
    ).flatten(0, 1)

    # careful: same global transformation for each jet
    random = rand_lorentz(batch_dims, dtype=dtype)
    random = random.unsqueeze(1).repeat(1, jet_size, 1, 1).view(*fm.shape, 4)

    # path 1: global transform + predict vectors
    fm_prime = torch.einsum("...ij,...j->...i", random, fm)
    node_attr_prime = calc_node_attr(fm_prime)
    vecs_prime1 = equivectors(fourmomenta=fm_prime, scalars=node_attr_prime, ptr=ptr)

    # path 2: predict vectors + global transform
    node_attr = calc_node_attr(fm)
    vecs = equivectors(fourmomenta=fm, scalars=node_attr, ptr=ptr)
    vecs_prime2 = torch.einsum("...ij,...kj->...ki", random, vecs)

    # test that vectors are predicted equivariantly
    torch.testing.assert_close(vecs_prime1, vecs_prime2, **TOLERANCES)
