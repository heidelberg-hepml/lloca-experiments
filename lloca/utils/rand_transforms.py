"""Random Lorentz transformations."""

import torch
from typing import List

from .lorentz import lorentz_eye
from .polar_decomposition import restframe_boost


def get_trafo_type(axis):
    """Determine whether the transformation is a boost or a rotation,
    based on the spacetime axes that are involved.

    Parameters
    ----------
    axis : torch.Tensor
        Axis of the transformation, should be of shape (2, ...)

    Returns
    -------
    trafo_type : torch.Tensor
        A boolean tensor indicating whether the transformation is a boost (1) or a rotation (0).
    """
    return torch.any(axis == 0, dim=0)


def transform(
    axes: List[int],
    angles: List[torch.Tensor],
    use_float64=True,
):
    """Recursively build transformation matrices based on given lists of axes and angles.
    This function is very flexible, but transformations built in this way suffer from
    numerical inaccuracies when many transformations are chained together.

    Parameters
    ----------
    axes : List[torch.Tensor]
        List of axes along which the transformations are performed.
        Each element is a tensor of shape (2, ...).
    angles : List[torch.Tensor]
        List of angles used for the transformations.
        Each element is a tensor of shape (...,).
    use_float64 : bool, optional
        Whether to use float64 for calculations, by default True

    Returns
    -------
    final_trafo : torch.Tensor
        Final transformation matrix of shape (..., 4, 4).
    """
    assert len(axes) == len(angles)
    dims = angles[0].shape
    assert all([angle.shape == dims for angle in angles])
    assert all([axis[0].shape == dims for axis in axes])
    assert all([axis[1].shape == dims for axis in axes])

    in_dtype = angles[0].dtype
    dtype = torch.float64 if use_float64 else in_dtype
    angles = [a.to(dtype=dtype) for a in angles]

    final_trafo = lorentz_eye(dims, angles[0].device, angles[0].dtype)
    for axis, angle in zip(axes, angles):
        trafo = lorentz_eye(dims, angle.device, angle.dtype)
        trafo_type = get_trafo_type(axis)

        meshgrid = torch.meshgrid(*[torch.arange(d) for d in dims], indexing="ij")
        trafo[(*meshgrid, axis[0], axis[0])] = torch.where(
            trafo_type, torch.cosh(angle), torch.cos(angle)
        )
        trafo[(*meshgrid, axis[0], axis[1])] = torch.where(
            trafo_type, torch.sinh(angle), -torch.sin(angle)
        )
        trafo[(*meshgrid, axis[1], axis[0])] = torch.where(
            trafo_type, torch.sinh(angle), torch.sin(angle)
        )
        trafo[(*meshgrid, axis[1], axis[1])] = torch.where(
            trafo_type, torch.cosh(angle), torch.cos(angle)
        )
        final_trafo = torch.einsum("...jk,...kl->...jl", trafo, final_trafo)
    return final_trafo.to(in_dtype)


def rand_lorentz(
    shape: List[int],
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """Create general Lorentz transformations as rotation * boost.
    Any Lorentz transformation can be expressed in this way,
    see polar decomposition of the Lorentz group.

    Parameters
    ----------
    shape: List[int]
        Shape of the transformation matrices
    std_eta: float
        Standard deviation of rapidity
    n_max_std_eta: float
        Allowed number of standard deviations;
        used to sample from a truncated Gaussian
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    assert std_eta > 0
    boost = rand_boost(
        shape,
        std_eta,
        n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    rotation = rand_rotation(shape, device, dtype, generator=generator)

    trafo = torch.einsum("...ij,...jk->...ik", rotation, boost)
    return trafo


def rand_xyrotation(
    shape: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """Create xy-plane rotation matrices embedded in the Lorentz group.

    Parameters
    ----------
    shape: List[int]
        Shape of the transformation matrices
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    axis = torch.tensor([1, 2], dtype=torch.long, device=device)
    axis = axis.view(2, *([1] * len(shape))).repeat(1, *shape)
    angle = (
        torch.rand(*shape, device=device, dtype=dtype, generator=generator)
        * 2
        * torch.pi
    )
    return transform([axis], [angle])


def rand_ztransform(
    shape: List[int],
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create Lorentz transformations consisting of a boost along
    the z-axis and a rotation around the z-axis.
    This transformation is common in LHC physics.

    Parameters
    ----------
    shape: List[int]
        Shape of the transformation matrices
    std_eta: float
        Standard deviation of rapidity
    n_max_std_eta: float
        Allowed number of standard deviations;
        used to sample from a truncated Gaussian
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    # rotation around z-axis
    axis1 = torch.tensor([1, 2], dtype=torch.long, device=device)
    axis1 = axis1.view(2, *([1] * len(shape))).repeat(1, *shape)
    angle1 = (
        torch.rand(*shape, device=device, dtype=dtype, generator=generator)
        * 2
        * torch.pi
    )

    # boost along z-axis
    axis2 = torch.tensor([0, 3], dtype=torch.long, device=device)
    axis2 = axis2.view(2, *([1] * len(shape))).repeat(1, *shape)
    angle2 = sample_rapidity(
        shape,
        std_eta=std_eta,
        n_max_std_eta=n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    return transform([axis1, axis2], [angle1, angle2])


def rand_rotation(
    shape: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """
    Create rotation matrices embedded in Lorentz transformations.
    The rotations are sampled uniformly using quaternions,
    see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.

    Parameters
    ----------
    shape: List[int]
        Shape of the transformation matrices
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    # generate random quaternions
    u = torch.rand(*shape, 3, device=device, dtype=dtype, generator=generator)
    q1 = torch.sqrt(1 - u[..., 0]) * torch.sin(2 * torch.pi * u[..., 1])
    q2 = torch.sqrt(1 - u[..., 0]) * torch.cos(2 * torch.pi * u[..., 1])
    q3 = torch.sqrt(u[..., 0]) * torch.sin(2 * torch.pi * u[..., 2])
    q0 = torch.sqrt(u[..., 0]) * torch.cos(2 * torch.pi * u[..., 2])

    # create rotation matrix from quaternions
    R1 = torch.stack(
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        dim=-1,
    )
    R2 = torch.stack(
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        dim=-1,
    )
    R3 = torch.stack(
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        dim=-1,
    )
    R = torch.stack([R1, R2, R3], dim=-2)

    trafo = torch.eye(4, device=device, dtype=dtype).expand(*shape, 4, 4).clone()
    trafo[..., 1:, 1:] = R
    return trafo


def rand_boost(
    shape: List[int],
    std_eta: float = 0.1,
    n_max_std_eta: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator = None,
):
    """Create a general pure boost, i.e. a symmetric Lorentz transformation.

    Parameters
    ----------
    shape: List[int]
        Shape of the transformation matrices
    std_eta: float
        Standard deviation of rapidity
    n_max_std_eta: float
        Allowed number of standard deviations;
        used to sample from a truncated Gaussian
    device: str
    dtype: torch.dtype
    generator: torch.Generator

    Returns
    -------
    final_trafo: torch.tensor
        The resulting Lorentz transformation matrices of shape (..., 4, 4).
    """
    shape = list(shape) + [3]
    beta = sample_rapidity(
        shape,
        std_eta,
        n_max_std_eta,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    beta2 = (beta**2).sum(dim=-1, keepdim=True)
    gamma = 1 / (1 - beta2).clamp(min=1e-10).sqrt()
    fourmomenta = torch.cat([gamma, beta], axis=-1)

    boost = restframe_boost(fourmomenta)
    return boost


def sample_rapidity(
    shape,
    std_eta,
    n_max_std_eta=3.0,
    device="cpu",
    dtype=torch.float32,
    generator: torch.Generator = None,
):
    """Sample rapidity from a clipped gaussian distribution.

    Parameters
    ----------
    shape: List[int]
        Shape of the output tensor
    std_eta: float
        Standard deviation of the rapidity
    n_max_std_eta: float
        Maximum number of standard deviations for truncation
    device: str
    dtype: torch.dtype
    generator: torch.Generator
    """
    angle = (
        torch.randn(*shape, device=device, dtype=dtype, generator=generator) * std_eta
    )
    truncate_mask = torch.abs(angle) > std_eta * n_max_std_eta
    while truncate_mask.any():
        new_angle = (
            torch.randn(*shape, device=device, dtype=dtype, generator=generator)
            * std_eta
        )
        angle[truncate_mask] = new_angle[truncate_mask]
        truncate_mask = torch.abs(angle) > std_eta * n_max_std_eta
    return angle
