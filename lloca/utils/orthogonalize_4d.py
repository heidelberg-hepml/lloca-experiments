"""Orthogonalization of Minkowski vectors."""

import torch

from .lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
    lorentz_cross,
)


def orthogonalize_4d(vecs, use_float64=True, return_reg=False, **kwargs):
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

    trafo = timelike_first(trafo)
    scale = trafo.new_tensor((1, -1, -1, -1))
    trafo = trafo * torch.outer(scale, scale)
    if use_float64:
        trafo = trafo.to(original_dtype)
    return (trafo, *reg) if return_reg else trafo


def orthogonalize_wrapper_4d(
    vecs,
    method="gramschmidt",
    eps_norm=1e-15,
    eps_reg_coplanar=1e-10,
    eps_reg_lightlike=1e-10,
    return_reg=False,
):
    """Wrapper for orthogonalization of Minkowski vectors.

    Parameters
    ----------
    vecs : torch.Tensor
        Tensor containing list of three Minkowski vectors of shape (..., 3, 4).
    method : str
        Method for orthogonalization. Options are "cross" and "gramschmidt".
    eps_norm : float
        Numerical regularization for the normalization of the vectors.
    eps_reg_coplanar : float
        Controls the scale of the regularization for coplanar vectors.
        eps_reg_coplanar**2 defines the selection threshold.
    eps_reg_lightlike : float
        Controls the scale of the regularization for lightlike vectors.
        eps_reg_lightlike**2 defines the selection threshold.
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
    vecs, reg_lightlike = regularize_lightlike(vecs, eps_reg_lightlike)
    vecs, reg_coplanar = regularize_coplanar(vecs, eps_reg_coplanar)

    if method == "cross":
        trafo = orthogonalize_cross(vecs, eps_norm)
    elif method == "gramschmidt":
        trafo = orthogonalize_gramschmidt(vecs, eps_norm)
    else:
        raise ValueError(f"Orthogonalization method {method} not implemented")

    return (trafo, reg_lightlike, reg_coplanar) if return_reg else trafo


def orthogonalize_gramschmidt(vecs, eps_norm=1e-15):
    """Gram-Schmidt orthogonalization algorithm for Minkowski vectors.

    Parameters
    ----------
    vecs : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4).
    eps_norm : float
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

    """
    v_nexts = [v for v in vecs]
    orthogonal_vecs = [vecs[0]]
    for i in range(1, len(vecs)):
        for k in range(i, len(vecs)):
            v_inner = lorentz_inner(v_nexts[k], orthogonal_vecs[i - 1]).unsqueeze(-1)
            v_norm = lorentz_squarednorm(orthogonal_vecs[i - 1]).unsqueeze(-1)
            v_nexts[k] = v_nexts[k] - orthogonal_vecs[i - 1] * v_inner / (
                v_norm + eps_norm
            )
        orthogonal_vecs.append(normalize_4d(v_nexts[i], eps_norm))
    last_vec = normalize_4d(lorentz_cross(*orthogonal_vecs), eps_norm)
    orthogonal_vecs.append(last_vec)

    return orthogonal_vecs
    """


def orthogonalize_cross(vecs, eps_norm=1e-15):
    """Orthogonalization algorithm using repeated cross products.
    This approach gives the same result as orthogonalize_gramschmidt for unlimited
    precision, but we find empirically that the Gram-Schmidt approach is more stable.

    Parameters
    ----------
    vecs : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4).
    eps_norm : float
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


def timelike_first(trafo):
    """Reorder the Lorentz transformation such that the first vector is timelike.
    This is necessary to ensure that the resulting Lorentz transformation has the
    correct metric signature (1, -1, -1, -1). Note that this step can be skipped
    if the first vector is already timelike.

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

    idx = (norm > 0).to(torch.long).argmax(dim=-1)
    base3 = torch.arange(3, device=trafo.device)
    i = idx.unsqueeze(-1)
    others = base3 + (base3 >= i)
    order = torch.cat([i, others], dim=-1)

    idx_rows = order.unsqueeze(-1).expand(*order.shape, trafo.size(-1))
    trafo_reordered = trafo.gather(dim=-2, index=idx_rows)
    return trafo_reordered


def regularize_lightlike(vecs, eps_reg_lightlike=1e-10):
    """If the Minkowski norm of a vector is close to zero,
    it is lightlike. In this case, we add a bit of noise to the vector
    to break the degeneracy and ensure that the orthogonalization works.

    Parameters
    ----------
    vecs : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4).
    eps_reg_lightlike : float
        Small value to control the scale of the regularization for lightlike vectors.

    Returns
    -------
    vecs_reg : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4) with regularization applied.
    reg_lightlike : int
        Number of vectors that were regularized due to being lightlike.
    """
    inners = lorentz_squarednorm(vecs)
    mask = inners.abs() < eps_reg_lightlike**2

    vecs_reg = vecs + mask.unsqueeze(-1) * eps_reg_lightlike * torch.randn_like(vecs)
    reg_lightlike = mask.any(dim=-1).sum()
    return vecs_reg, reg_lightlike


def regularize_coplanar(vecs, eps_reg_coplanar=1e-10):
    """If the cross product of three vectors is close to zero,
    they are coplanar. In this case, we add a bit of noise to each vector
    to break the degeneracy and ensure that the orthogonalization works.

    Parameters
    ----------
    vecs : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4).
    eps_reg_coplanar : float
        Small value to control the scale of the regularization for coplanar vectors.

    Returns
    -------
    vecs_reg : torch.Tensor
        List of three Minkowski vectors of shape (..., 3, 4) with regularization applied.
    reg_coplanar : int
        Number of vectors that were regularized due to coplanarity.
    """
    v0, v1, v2 = vecs.unbind(dim=-2)
    cross_norm2 = lorentz_squarednorm(lorentz_cross(v0, v1, v2))
    mask = cross_norm2.abs() < eps_reg_coplanar**2

    vecs_reg = vecs + mask.unsqueeze(-1).unsqueeze(
        -1
    ) * eps_reg_coplanar * torch.randn_like(vecs)
    reg_coplanar = mask.sum()
    return vecs_reg, reg_coplanar


def normalize_4d(v, eps=1e-15):
    """Normalize a Minkowski vector by the absolute value of the Minkowski norm.
    Note that this norm can be close to zero.

    Parameters
    ----------
    v : torch.Tensor
        Minkowski vector of shape (..., 4).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized Minkowski vector of shape (..., 4).
    """
    norm = lorentz_squarednorm(v).unsqueeze(-1)
    norm = norm.abs().sqrt()
    return v / (norm + eps)
