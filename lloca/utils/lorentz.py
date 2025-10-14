"""Basic operations in Minkowski space."""

import torch


def lorentz_inner(v1, v2):
    """Lorentz inner product, i.e v1^T @ g @ v2

    Parameters
    ----------
    v1, v2 : torch.Tensor
        Tensors of shape (..., 4)

    Returns
    -------
    torch.Tensor
        Lorentz inner product of shape (..., )
    """
    t = v1[..., 0] * v2[..., 0]
    s = (v1[..., 1:] * v2[..., 1:]).sum(dim=-1)
    return t - s


def lorentz_squarednorm(v):
    """Lorentz norm, i.e. v^T @ g @ v

    Parameters
    ----------
    v : torch.Tensor
        Tensor of shape (..., 4)

    Returns
    -------
    torch.Tensor
        Lorentz norm of shape (..., )
    """
    return lorentz_inner(v, v)


def lorentz_eye(dims, device=torch.device("cpu"), dtype=torch.float32):
    """
    Create a identity matrix of given shape

    Parameters
    ----------
    dims : tuple
        Dimension of the output tensor, e.g. (2, 3) for a 2x3 matrix
    device : torch.device
        Device to create the tensor on, by default torch.device("cpu")
    dtype : torch.dtype
        Data type of the tensor, by default torch.float32

    Returns
    -------
    torch.Tensor
        Identity matrix of shape (..., 4, 4)
    """
    base_eye = torch.eye(4, dtype=dtype, device=device)
    eye = base_eye.view((1,) * len(dims) + (4, 4)).expand(*dims, 4, 4)
    return eye


def lorentz_metric(dims, device=torch.device("cpu"), dtype=torch.float32):
    """
    Create a metric tensor of given shape

    Parameters
    ----------
    dims : tuple
        Dimension of the output tensor, e.g. (2, 3) for a 2x3 matrix
    device : torch.device
        Device to create the tensor on, by default torch.device("cpu")
    dtype : torch.dtype
        Data type of the tensor, by default torch.float32

    Returns
    -------
    torch.Tensor
        Metric tensor of shape (..., 4, 4)
    """
    diag = torch.tensor((1, -1, -1, -1), device=device, dtype=dtype)
    M = torch.diag_embed(diag)
    M = M.reshape((1,) * len(dims) + (4, 4)).expand(*dims, 4, 4)
    return M


def lorentz_cross(v1, v2, v3):
    """
    Compute the cross product in Minkowski space using the Laplace expansion.
    Note that this cross product takes three inputs vectors.

    Parameters
    ----------
    v1, v2, v3 : torch.Tensor
        Tensors of shape (..., 4) representing vectors in Minkowski space.

    Returns
    -------
    torch.Tensor
        The cross product of the three vectors, shape (..., 4).
    """
    assert v1.shape[-1] == 4
    assert v1.shape == v2.shape and v1.shape == v3.shape

    mat = torch.stack([v1, v2, v3], dim=-1)

    # euclidean fully antisymmetric product
    v4 = []
    for n in range(4):
        minor = torch.cat([mat[..., :n, :], mat[..., n + 1 :, :]], dim=-2)
        contribution = (-1) ** n * torch.det(minor)
        v4.append(contribution)
    v4 = torch.stack(v4, dim=-1)

    # raise indices with metric tensor
    v4 *= torch.tensor([1.0, -1.0, -1.0, -1.0], device=v1.device, dtype=v1.dtype)
    return v4
