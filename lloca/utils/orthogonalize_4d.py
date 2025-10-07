"""Orthogonalization of Minkowski vectors."""

import torch

from .lorentz import (
    lorentz_inner,
    lorentz_squarednorm,
    lorentz_metric,
    lorentz_cross,
)


def orthogonalize_4d(vecs, use_float64=True, return_reg=False, **kwargs):
    """High-level wrapper for orthogonalization of three Minkowski vectors.

    Parameters
    ----------
    vecs : list of torch.Tensor
        List of three Minkowski vectors of shape (..., 4).
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
        vecs = [v.to(torch.float64) for v in vecs]

    out = orthogonalize_wrapper_4d(vecs, return_reg=return_reg, **kwargs)
    if return_reg:
        orthogonal_vecs, *reg = out
    else:
        orthogonal_vecs = out
    trafo = torch.stack(orthogonal_vecs, dim=-2)

    trafo = timelike_first(trafo)
    metric = lorentz_metric(trafo.shape[:-2], device=trafo.device, dtype=trafo.dtype)
    trafo = metric @ trafo @ metric
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
    vecs : list of torch.Tensor
        List of three Minkowski vectors of shape (..., 4).
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
    orthogonal_vecs : list of torch.Tensor
        List of orthogonalized Minkowski vectors of shape (..., 4).
    reg_lightlike : int
        Number of vectors that were regularized due to being lightlike.
    reg_coplanar : int
        Number of vectors that were regularized due to coplanarity.
    """
    assert len(vecs) == 3
    assert all(v.shape == vecs[0].shape for v in vecs)

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
    vecs : list of torch.Tensor
        List of Minkowski vectors of shape (..., 4).
    eps_norm : float
        Small value to avoid division by zero during normalization.

    Returns
    -------
    orthogonal_vecs : list of torch.Tensor
        List of orthogonalized Minkowski vectors of shape (..., 4).
    """
    vecs = [normalize_4d(v, eps_norm) for v in vecs]

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


def orthogonalize_cross(vecs, eps_norm=1e-15):
    """Orthogonalization algorithm using repeated cross products.
    This approach gives the same result as orthogonalize_gramschmidt for unlimited
    precision, but we find empirically that the Gram-Schmidt approach is more stable.

    Parameters
    ----------
    vecs : list of torch.Tensor
        List of Minkowski vectors of shape (..., 4).
    eps_norm : float
        Small value to avoid division by zero during normalization.

    Returns
    -------
    orthogonal_vecs : list of torch.Tensor
        List of orthogonalized Minkowski vectors of shape (..., 4).
    """
    vecs = [normalize_4d(v, eps_norm) for v in vecs]

    orthogonal_vecs = [vecs[0]]
    for i in range(1, len(vecs) + 1):
        v_next = lorentz_cross(*orthogonal_vecs, *vecs[i:])
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize_4d(v_next, eps_norm))

    return orthogonal_vecs


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
    pos_norm = norm > 0
    num_pos_norm = pos_norm.sum(dim=-1)
    assert (
        num_pos_norm == 1
    ).all(), f"Don't always have exactly 1 timelike vector: {(num_pos_norm==0).sum().item()} (#0), {(num_pos_norm==1).sum().item()} (#1), {(num_pos_norm==2).sum().item()} (#2), {(num_pos_norm==3).sum().item()} (#3)"
    old_trafo = trafo.clone()
    trafo[..., 0, :] = old_trafo[pos_norm].view(*trafo.shape[:-2], 4)
    trafo[..., 1:, :] = old_trafo[~pos_norm].view(*trafo.shape[:-2], 3, 4)
    return trafo


def regularize_lightlike(vecs, eps_reg_lightlike=1e-10):
    """If the Minkowski norm of a vector is close to zero,
    it is lightlike. In this case, we add a bit of noise to the vector
    to break the degeneracy and ensure that the orthogonalization works.

    Parameters
    ----------
    vecs : list of torch.Tensor
        List of Minkowski vectors of shape (..., 4).
    eps_reg_lightlike : float
        Small value to control the scale of the regularization for lightlike vectors.

    Returns
    -------
    vecs_reg : list of torch.Tensor
        List of Minkowski vectors of shape (..., 4) with regularization applied.
    reg_lightlike : int
        Number of vectors that were regularized due to being lightlike.
    """
    vecs_reg = []
    masks = []
    for v in vecs:
        inners = lorentz_inner(v, v)
        mask = inners.abs() < eps_reg_lightlike**2
        v_reg = v + eps_reg_lightlike * torch.randn_like(v) * mask.unsqueeze(-1)
        masks.append(mask)
        vecs_reg.append(v_reg)

    reg_lightlike = torch.stack(masks).any(dim=-1).sum().item()
    return vecs_reg, reg_lightlike


def regularize_coplanar(vecs, eps_reg_coplanar=1e-10):
    """If the cross product of three vectors is close to zero,
    they are coplanar. In this case, we add a bit of noise to each vector
    to break the degeneracy and ensure that the orthogonalization works.

    Parameters
    ----------
    vecs : list of torch.Tensor
        List of three Minkowski vectors of shape (..., 4).
    eps_reg_coplanar : float
        Small value to control the scale of the regularization for coplanar vectors.

    Returns
    -------
    vecs_reg : list of torch.Tensor
        List of three Minkowski vectors of shape (..., 4) with regularization applied.
    reg_coplanar : int
        Number of vectors that were regularized due to coplanarity.
    """
    assert len(vecs) == 3
    cross_norm = lorentz_squarednorm(lorentz_cross(*vecs))
    mask = cross_norm.abs() < eps_reg_coplanar**2

    vecs_reg = []
    for v in vecs:
        v_reg = v + eps_reg_coplanar * torch.randn_like(v) * mask.unsqueeze(-1)
        vecs_reg.append(v_reg)

    reg_coplanar = mask.sum().item()
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
