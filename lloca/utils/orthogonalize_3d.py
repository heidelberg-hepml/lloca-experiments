"""Orthogonalization of euclidean vectors."""

import torch
import torch.nn.functional as F


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
    vecs : torch.Tensor
        Two vectors of shape (..., 2, 3).
    eps_norm : float
        Numerical regularization for the normalization of the vectors.

    Returns
    -------
    orthogonal_vecs : torch.Tensor
        Three orthogonalized vectors of shape (..., 3, 3),
        where dim=-2 counts the vectors.
    """
    vecs = F.normalize(vecs, dim=-1, eps=eps_norm)
    e0, v1 = vecs.unbind(dim=-2)

    u1 = v1 - (v1 * e0).sum(dim=-1, keepdim=True) * e0
    e1 = F.normalize(u1, dim=-1, eps=eps_norm)

    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-2)


def orthogonalize_cross_3d(vecs, eps_norm=1e-15):
    """Cross product orthogonalization algorithm for euclidean vectors.
    This approach is equivalent to the Gram-Schmidt procedure for unlimited precision,
    but for limited precision it is more stable.

    Parameters
    ----------
    vecs : torch.Tensor
        Two vectors of shape (..., 2, 3).
    eps_norm : float
        Numerical regularization for the normalization of the vectors.

    Returns
    -------
    orthogonal_vecs : torch.Tensor
        Three orthogonalized vectors of shape (..., 3, 3),
        where dim=-2 counts the vectors.
    """
    vecs = F.normalize(vecs, dim=-1, eps=eps_norm)
    e0, v1 = vecs.unbind(dim=-2)

    u1 = torch.cross(e0, v1)
    e1 = F.normalize(u1, dim=-1, eps=eps_norm)

    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-2)


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
    v0, v1 = vecs.unbind(dim=-2)
    cross = torch.cross(v0, v1, dim=-1)
    mask = (cross**2).sum(dim=-1) < eps_reg**4
    v1_reg = torch.where(mask.unsqueeze(-1), v1 + eps_reg * torch.randn_like(v1), v1)
    vecs_reg = torch.stack([v0, v1_reg], dim=-2)

    reg_collinear = mask.sum()
    return vecs_reg, reg_collinear
