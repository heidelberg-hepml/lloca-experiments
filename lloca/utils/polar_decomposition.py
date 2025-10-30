"""Lorentz transformations from boosts and rotations."""

import torch

from .lorentz import lorentz_squarednorm
from .orthogonalize_4d import orthogonalize_4d, regularize_lightlike


def restframe_boost(fourmomenta):
    """Construct a Lorentz transformation that boosts four-momenta into their rest frame.

    Parameters
    ----------
    fourmomenta : torch.Tensor
        Tensor of shape (..., 4) representing the four-momenta.

    Returns
    -------
    trafo : torch.Tensor
        Tensor of shape (..., 4, 4) representing the Lorentz transformation
        that boosts the four-momenta into their rest frame.
    """
    assert (
        lorentz_squarednorm(fourmomenta) > 0
    ).all(), "Trying to boost spacelike vectors into their restframe (not possible). Consider changing the nonlinearity in equivectors."

    # compute relevant quantities
    t0 = fourmomenta.narrow(-1, 0, 1)
    beta = fourmomenta[..., 1:] / t0.clamp_min(1e-10)
    beta2 = beta.square().sum(dim=-1, keepdim=True)
    one_minus_beta2 = torch.clamp_min(1 - beta2, min=1e-10)
    gamma = torch.rsqrt(one_minus_beta2)
    boost = -gamma * beta

    # prepare rotation part
    eye3 = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye3 = eye3.reshape(*(1,) * len(fourmomenta.shape[:-1]), 3, 3).expand(
        *fourmomenta.shape[:-1], 3, 3
    )
    scale = (gamma - 1) / torch.clamp_min(beta2, min=1e-10)
    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)
    rot = eye3 + scale.unsqueeze(-1) * outer

    # collect trafo
    row0 = torch.cat((gamma, boost), dim=-1)
    lower = torch.cat((boost.unsqueeze(-1), rot), dim=-1)
    trafo = torch.cat((row0.unsqueeze(-2), lower), dim=-2)
    return trafo


def polar_decomposition(fourmomenta, references, return_reg=False, **kwargs):
    """Construct a Lorentz transformation as a polar decomposition of a
    boost and a rotation.

    Parameters
    ----------
    fourmomenta : torch.Tensor
        Tensor of shape (..., 4) representing the four-momenta that define the rest frames.
    references : torch.Tensor
        Two tensors of shape (..., 2, 4) representing the reference four-momenta to construct the rotation.
    return_reg : bool
        If True, return a tuple with the Lorentz transformation and regularization information.
    kwargs : dict
        It contains use_float64 (bool): If True, computations are done in float64 precision.
        And orthogonalization kwargs passed to `orthogonalize_4d`.

    Returns
    -------
    trafo : torch.Tensor
        Tensor of shape (..., 4, 4) representing the Lorentz transformation.
    reg_collinear : torch.Tensor, optional
        Tensor indicating if the references were collinear (only returned if `return_reg` is True).
    """
    assert fourmomenta.shape[:-1] == references.shape[:-2]

    use_float64 = kwargs.pop("use_float64", True)
    if use_float64:
        original_dtype = fourmomenta.dtype
        fourmomenta = fourmomenta.to(torch.float64)
        references = references.to(torch.float64)

    # regularize lightlike fourmomenta
    eps_reg_lightlike = kwargs.get("eps_reg_lightlike", None)
    if eps_reg_lightlike is not None:
        fourmomenta, reg_lightlike_1 = regularize_lightlike(
            fourmomenta, eps_reg_lightlike
        )
    else:
        fourmomenta, reg_lightlike_1 = regularize_lightlike(fourmomenta)

    # construct rest frame transformation
    boost = restframe_boost(fourmomenta)

    # references and fm go into rest frame
    fm_rest = torch.matmul(fourmomenta.unsqueeze(-2), boost.transpose(-1, -2))
    ref_rest = torch.matmul(references, boost.transpose(-1, -2))

    # construct rotation before orthogonalization
    inp = torch.cat((fm_rest, ref_rest), dim=-2)
    out = orthogonalize_4d(inp, return_reg=return_reg, **kwargs)
    if return_reg:
        rotation, reg_lighlike_2, reg_collinear = out
    else:
        rotation = out

    # combine rotation and boost
    trafo = torch.matmul(rotation, boost)
    if use_float64:
        trafo = trafo.to(original_dtype)
    return (
        (trafo, reg_lightlike_1 + reg_lighlike_2, reg_collinear)
        if return_reg
        else trafo
    )
