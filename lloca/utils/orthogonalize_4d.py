"""Orthogonalization of Minkowski vectors."""

import torch

from .lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
    lorentz_cross,
)


def orthogonalize_4d(vecs, use_float64=True, return_reg=False, checks=False, **kwargs):
    """High-level wrapper for orthogonalization of three Minkowski vectors.

    Parameters
    ----------
    vecs : torch.Tensor
        Tensor containing three Minkowski vectors of shape (..., 3, 4).
    use_float64 : bool
        If True, use float64 for numerical stability during orthogonalization.
    return_reg : bool
        If True, return a tuple with the orthogonalized vectors and the number of
        regularized vectors for lightlike and coplanar cases.
    checks : bool
        If True, perform additional assertion checks on predicted vectors.
        It may cause slowdowns due to GPU/CPU synchronization, use only for debugging.
    kwargs : dict
        Additional keyword arguments passed to the orthogonalization function.

    Returns
    -------
    trafo : torch.Tensor
        Lorentz transformation of shape (..., 4, 4) that orthogonalizes the input vectors.
        The first vector is guaranteed to be timelike.
    reg_lightlike : int
        Number of vectors that were regularized due to being lightlike.
    reg_coplanar : int
        Number of vectors that were regularized due to coplanarity.
    """
    if use_float64:
        original_dtype = vecs[0].dtype
        vecs = vecs.to(torch.float64)

    out = orthogonalize_wrapper_4d(vecs, return_reg=return_reg, **kwargs)
    if return_reg:
        orthogonal_vecs, *reg = out
    else:
        orthogonal_vecs = out
    trafo = orthogonal_vecs
    if checks:
        check_timelike_first(trafo)

    scale = trafo.new_tensor((1, -1, -1, -1))
    trafo = trafo * torch.outer(scale, scale)
    if use_float64:
        trafo = trafo.to(original_dtype)
    return (trafo, *reg) if return_reg else trafo


def orthogonalize_wrapper_4d(
    vecs,
    method="gramschmidt",
    eps_norm=None,
    eps_reg_coplanar=None,
    eps_reg_lightlike=None,
    return_reg=False,
):
    """Wrapper for orthogonalization of Minkowski vectors.

    Parameters
    ----------
    vecs : torch.Tensor
        Tensor containing list of three Minkowski vectors of shape (..., 3, 4).
    method : str
        Method for orthogonalization. Options are "cross" and "gramschmidt".
    eps_norm : float or None
        Numerical regularization for the normalization of the vectors.
        If None, use the smallest representable value for the vectors dtype.
    eps_reg_coplanar : float or None
        Controls the scale of the regularization for coplanar vectors.
        eps_reg_coplanar defines the selection threshold.
    eps_reg_lightlike : float or None
        Controls the scale of the regularization for lightlike vectors.
        eps_reg_lightlike defines the selection threshold.
    return_reg : bool
        If True, return a tuple with the orthogonalized vectors and the number of
        regularized vectors for lightlike and coplanar cases.

    Returns
    -------
    orthogonal_vecs : torch.Tensor
        Four orthogonalized Minkowski vectors of shape (..., 4, 4).
    reg_lightlike : int
        Number of vectors that were regularized due to being lightlike.
    reg_coplanar : int
        Number of vectors that were regularized due to coplanarity.
    """
    eps_norm = torch.finfo(vecs.dtype).eps if eps_norm is None else eps_norm

    vecs, reg_lightlike = regularize_lightlike(vecs, eps_reg_lightlike)
    vecs, reg_coplanar = regularize_coplanar(vecs, eps_reg_coplanar)

    if method == "cross":
        trafo = orthogonalize_cross(vecs, eps_norm)
    elif method == "gramschmidt":
        trafo = orthogonalize_gramschmidt(vecs, eps_norm)
    else:
        raise ValueError(f"Orthogonalization method {method} not implemented")

    return (trafo, reg_lightlike, reg_coplanar) if return_reg else trafo


def orthogonalize_gramschmidt(vecs, eps_norm=None):
    """Gram-Schmidt orthogonalization algorithm for Minkowski vectors.

    Parameters
    ----------
    vecs : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4).
    eps_norm : float or None
        Small value to avoid division by zero during normalization.

    Returns
    -------
    orthogonal_vecs : torch.Tensor
        List of four orthogonalized Minkowski vectors of shape (..., 4, 4).
    """
    vecs = normalize_4d(vecs, eps_norm)
    e0, v1, v2 = vecs.unbind(dim=-2)

    denom0 = lorentz_squarednorm(e0).unsqueeze(-1) + eps_norm
    inner01 = lorentz_inner(v1, e0).unsqueeze(-1)
    u1 = v1 - e0 * inner01 / denom0
    e1 = normalize_4d(u1, eps_norm)

    inner02 = lorentz_inner(v2, e0).unsqueeze(-1)
    u2 = v2 - e0 * inner02 / denom0
    denom1 = lorentz_squarednorm(e1).unsqueeze(-1) + eps_norm
    inner21 = lorentz_inner(v2, e1).unsqueeze(-1)
    u2 = u2 - e1 * inner21 / denom1
    e2 = normalize_4d(u2, eps_norm)

    e3 = lorentz_cross(e0, e1, e2)
    return torch.stack([e0, e1, e2, e3], dim=-2)


def orthogonalize_cross(vecs, eps_norm=None):
    """Orthogonalization algorithm using repeated cross products.
    This approach gives the same result as orthogonalize_gramschmidt for unlimited
    precision, but we find empirically that the Gram-Schmidt approach is more stable.

    Parameters
    ----------
    vecs : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4).
    eps_norm : float or None
        Small value to avoid division by zero during normalization.

    Returns
    -------
    orthogonal_vecs : torch.Tensor
        List of four orthogonalized Minkowski vectors of shape (..., 4, 4).
    """
    vecs = normalize_4d(vecs, eps_norm)
    e0, v1, v2 = vecs.unbind(dim=-2)

    e1 = normalize_4d(lorentz_cross(e0, v1, v2), eps_norm)
    e2 = normalize_4d(lorentz_cross(e0, e1, v2), eps_norm)
    e3 = normalize_4d(lorentz_cross(e0, e1, e2), eps_norm)
    return torch.stack([e0, e1, e2, e3], dim=-2)


def check_timelike_first(trafo):
    """Check that the Lorentz transformation has only one timelike vector.
    This is necessary to ensure that the resulting Lorentz transformation has the
    correct metric signature (1, -1, -1, -1). Note that the timelike nature of the
    first vector is enforced by the nonlinearity in the framesnet, this will trigger
    if numerical instabilities occur in the orthogonalization.

    Parameters
    ----------
    trafo : torch.Tensor
        Lorentz transformation of shape (..., 4, 4) where the last two dimensions
        represent the transformation matrix.

    Returns
    -------
    trafo : torch.Tensor
        Lorentz transformation of shape (..., 4, 4) with the first vector being timelike.
    """
    vecs = [trafo[..., i, :] for i in range(4)]
    norm = torch.stack([lorentz_squarednorm(v) for v in vecs], dim=-1)
    num_pos_norm = (norm > 0).sum(dim=-1)
    assert (num_pos_norm == 1).all(), "Don't always have exactly 1 timelike vector"


def regularize_lightlike(vecs, eps_reg_lightlike=None):
    """If the Minkowski norm of a vector is close to zero,
    it is lightlike. In this case, we add a bit of noise to the vector
    to break the degeneracy and ensure that the orthogonalization works.
    The noise is sampled such that the resulting vector is timelike.

    Parameters
    ----------
    vecs : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4).
    eps_reg_lightlike : float or None
        Small value to control the scale of the regularization for lightlike vectors.
        If None, use the smallest representable value for the vectors dtype.

    Returns
    -------
    vecs_reg : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4) with regularization applied.
    reg_lightlike : int
        Number of vectors that were regularized due to being lightlike.
    """
    eps_reg_lightlike = (
        torch.finfo(vecs.dtype).eps if eps_reg_lightlike is None else eps_reg_lightlike
    )
    inners = lorentz_squarednorm(vecs)
    mask = inners.abs() < eps_reg_lightlike

    # calculate the 3-norm and set the 0th-component accordingly
    randn_vecs = torch.randn_like(vecs).abs()
    randn_vecs_3sqnorm = (randn_vecs[..., 1:] ** 2).sum(dim=-1)
    randn_vecs[..., 0] = (
        2 * randn_vecs_3sqnorm
    ).sqrt()  # heuristic factor of 2 to ensure timelike

    vecs_reg = vecs + mask.unsqueeze(-1) * eps_reg_lightlike * randn_vecs
    reg_lightlike = mask.any(dim=-1).sum()
    return vecs_reg, reg_lightlike


def regularize_coplanar(vecs, eps_reg_coplanar=None):
    """If the cross product of three vectors is close to zero,
    they are coplanar. In this case, we add a bit of noise to each vector
    to break the degeneracy and ensure that the orthogonalization works.

    Parameters
    ----------
    vecs : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4).
    eps_reg_coplanar : float or None
        Small value to control the scale of the regularization for coplanar vectors.
        If None, use the smallest representable value for the vectors dtype.

    Returns
    -------
    vecs_reg : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4) with regularization applied.
    reg_coplanar : int
        Number of vectors that were regularized due to coplanarity.
    """
    eps_reg_coplanar = (
        torch.finfo(vecs.dtype).eps if eps_reg_coplanar is None else eps_reg_coplanar
    )
    v0, v1, v2 = vecs.unbind(dim=-2)
    cross_norm2 = lorentz_squarednorm(lorentz_cross(v0, v1, v2))
    mask = cross_norm2.abs() < eps_reg_coplanar

    vecs_reg = vecs + mask.unsqueeze(-1).unsqueeze(
        -1
    ) * eps_reg_coplanar * torch.randn_like(vecs)
    reg_coplanar = mask.sum()
    return vecs_reg, reg_coplanar


def normalize_4d(v, eps=None):
    """Normalize a Minkowski vector by the absolute value of the Minkowski norm.
    Note that this norm can be close to zero.

    Parameters
    ----------
    v : torch.Tensor
        Minkowski vector of shape (..., 4).
    eps : float or None
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized Minkowski vector of shape (..., 4).
    """
    norm = lorentz_squarednorm(v).unsqueeze(-1)
    norm = norm.abs().sqrt()
    return v / (norm + eps)
