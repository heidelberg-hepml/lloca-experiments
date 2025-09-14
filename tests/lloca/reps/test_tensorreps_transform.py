import pytest
import torch
from tests.constants import TOLERANCES, BATCH_DIMS

from lloca.utils.rand_transforms import rand_lorentz
from lloca.framesnet.frames import Frames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform


@pytest.mark.parametrize("use_naive", [True, False])
@pytest.mark.parametrize("batch_dim", BATCH_DIMS)
def test_tensorreps(batch_dim, use_naive):
    frames = rand_lorentz(batch_dim)
    frames = Frames(frames)

    rep = "5x0n+3x1n+3x2n+5x3n"
    rep = TensorReps(rep)
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    tensor_reps_transform(coeffs, frames)

    # 0n test
    rep = TensorReps("5x0n")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    torch.testing.assert_close(transformed_coeffs, coeffs, **TOLERANCES)

    # 0p test
    rep = TensorReps("5x0p")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    torch.testing.assert_close(
        transformed_coeffs, coeffs * frames.det[..., None], **TOLERANCES
    )

    # 1n test
    rep = TensorReps("5x1n")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    torch.testing.assert_close(
        transformed_coeffs.reshape(*batch_dim, rep.max_rep.mul, 4),
        torch.matmul(
            coeffs.reshape(*batch_dim, rep.max_rep.mul, 4),
            frames.matrices.transpose(-1, -2),
        ),
        **TOLERANCES,
    )

    # 1p test
    rep = TensorReps("5x1p")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    torch.testing.assert_close(
        transformed_coeffs.reshape(*batch_dim, rep.max_rep.mul, 4),
        torch.matmul(
            coeffs.reshape(*batch_dim, rep.max_rep.mul, 4),
            frames.matrices.transpose(-1, -2),
        )
        * frames.det[..., None, None],
        **TOLERANCES,
    )

    # 2n test
    rep = TensorReps("5x2n")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    naive_trafo = torch.einsum(
        "...ij, ...lm, ...cjm -> ...cil",
        frames.matrices,
        frames.matrices,
        coeffs.reshape(*batch_dim, rep.max_rep.mul, 4, 4),
    )
    torch.testing.assert_close(
        transformed_coeffs.reshape(*batch_dim, rep.max_rep.mul, 4, 4),
        naive_trafo,
        **TOLERANCES,
    )

    # 2p test
    rep = TensorReps("5x2p")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    naive_trafo = torch.einsum(
        "...ij, ...lm, ...cjm -> ...cil",
        frames.matrices,
        frames.matrices,
        coeffs.reshape(*batch_dim, rep.max_rep.mul, 4, 4),
    )
    naive_trafo *= frames.det[..., None, None, None]
    torch.testing.assert_close(
        transformed_coeffs.reshape(*batch_dim, rep.max_rep.mul, 4, 4),
        naive_trafo,
        **TOLERANCES,
    )

    # 1x0n+1x2n test
    rep = TensorReps("1x0n+1x2n")
    coeffs = torch.randn(batch_dim + [rep.dim])
    tensor_reps_transform = TensorRepsTransform(rep)
    transformed_coeffs = tensor_reps_transform(coeffs.clone(), frames)
    naive_trafo = torch.einsum(
        "...ij, ...lm, ...cjm -> ...cil",
        frames.matrices,
        frames.matrices,
        coeffs[..., 1:].reshape(*batch_dim, rep.max_rep.mul, 4, 4),
    )
    torch.testing.assert_close(
        transformed_coeffs[..., 1:].reshape(*batch_dim, rep.max_rep.mul, 4, 4),
        naive_trafo,
        **TOLERANCES,
    )
    torch.testing.assert_close(
        coeffs[..., :1], transformed_coeffs[..., :1], **TOLERANCES
    )
