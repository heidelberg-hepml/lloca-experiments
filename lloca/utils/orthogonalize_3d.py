"""Orthogonalization of euclidean vectors."""

import torch


def orthogonalize_3d(
    vecs, method="gramschmidt", eps_norm=1e-15, eps_reg=1e-10, return_reg=False
):
    """Wrapper for orthogonalization of euclidean vectors.

    Parameters
    ----------
    vecs : list of torch.Tensor
        List of torch.tensor of shape (..., 3)
        Vectors to be orthogonalized
    method : str
        Method for orthogonalization. Options are "cross" and "gramschmidt".
    eps_norm : float
        Numerical regularization for the normalization of the vectors.
    eps_reg : float
        Controls the scale of the regularization for collinear vectors.
        eps_reg**2 defines the selection threshold.
    return_reg : bool
        If True, additionally return the number of regularized vectors for collinearity.

    Returns
    -------
    orthogonal_vecs : list of torch.Tensor
        List of orthogonalized vectors of shape (..., 3)
    reg_collinear : int
        Number of vectors that were regularized due to collinearity.
    """
    vecs, reg_collinear = regularize_collinear(vecs, eps_reg)

    if method == "cross":
        trafo = orthogonalize_cross_3d(vecs, eps_norm)
    elif method == "gramschmidt":
        trafo = orthogonalize_gramschmidt_3d(vecs, eps_norm)
    else:
        raise ValueError(f"Orthogonalization method {method} not implemented")

    return (trafo, reg_collinear) if return_reg else trafo


def orthogonalize_gramschmidt_3d(vecs, eps_norm=1e-15):
    """Gram-Schmidt orthogonalization algorithm for euclidean vectors.

    Parameters
    ----------
    vecs : list of torch.Tensor
        List of two vectors of shape (..., 3).
    eps_norm : float
        Numerical regularization for the normalization of the vectors.

    Returns
    -------
    orthogonal_vecs : list of torch.Tensor
        List of orthogonalized vectors of shape (..., 3).
    """
    n_vectors = len(vecs)
    assert n_vectors == 2

    vecs = [normalize_3d(v, eps_norm) for v in vecs]

    v_nexts = [v for v in vecs]
    orthogonal_vecs = [vecs[0]]

    # gram schmidt procedure
    for i in range(1, n_vectors):
        for k in range(i, n_vectors):
            v_inner = torch.sum(
                v_nexts[k] * orthogonal_vecs[i - 1], dim=-1, keepdim=True
            )
            v_nexts[k] = v_nexts[k] - orthogonal_vecs[i - 1] * v_inner
        orthogonal_vecs.append(normalize_3d(v_nexts[i], eps_norm))

    # last vector from cross product
    last_vec = torch.cross(*orthogonal_vecs, dim=-1)
    orthogonal_vecs.append(normalize_3d(last_vec, eps_norm))

    return orthogonal_vecs


def orthogonalize_cross_3d(vecs, eps_norm=1e-15):
    """Cross product orthogonalization algorithm for euclidean vectors.
    This approach is equivalent to the Gram-Schmidt procedure for unlimited precision,
    but for limited precision it is more stable.

    Parameters
    ----------
    vecs : list of torch.Tensor
        List of two vectors of shape (..., 3).
    eps_norm : float
        Numerical regularization for the normalization of the vectors.

    Returns
    -------
    orthogonal_vecs : list of torch.Tensor
        List of three orthogonalized vectors of shape (..., 3).
    """
    n_vectors = len(vecs)
    assert n_vectors == 2

    vecs = [normalize_3d(v, eps_norm) for v in vecs]

    orthogonal_vecs = [vecs[0]]
    for i in range(1, n_vectors + 1):
        v_next = torch.cross(*orthogonal_vecs, *vecs[i:], dim=-1)
        assert torch.isfinite(v_next).all()
        orthogonal_vecs.append(normalize_3d(v_next, eps_norm))

    return orthogonal_vecs


def regularize_collinear(vecs, eps_reg=1e-10):
    """If the cross product of two vectors is small, the vectors are collinear.
    In this case, we add a small amount of noise to the second vector to
    regularize the orthogonalization.

    Parameters
    ----------
    vecs : list of torch.Tensor
        List with 2 vectors of shape (..., 3).
    eps_reg : float
        Regularization epsilon, controls the scale of the noise added to the second vector.

    Returns
    -------
    vecs : list of torch.Tensor
        List with 2 vectors of shape (..., 3), where the second vector is regularized if collinear.
    reg_collinear : int
        Number of vectors that were regularized due to collinearity.
    """
    assert len(vecs) == 2
    mask = torch.linalg.norm(torch.cross(*vecs, dim=-1)).abs() < eps_reg**2
    vecs[1][mask] += eps_reg * torch.randn_like(vecs[1][mask])

    reg_collinear = mask.sum().item()
    return vecs, reg_collinear


def normalize_3d(v, eps_norm=1e-15):
    """Normalize an euclidean vector with numerical stability.

    Parameters
    ----------
    v : torch.Tensor
        A tensor of shape (..., 3) representing the vector to be normalized.
    eps_norm : float, optional
        A small value to prevent division by zero, by default 1e-10.

    Returns
    -------
    torch.Tensor
        The normalized vector of shape (..., 3).
    """
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / (norm + eps_norm)
