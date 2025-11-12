import math

import torch
from torch import nn

# log(x) -> log(x+EPS1)
# in (invertible) preprocessing functions to avoid being close to log(0)
EPS1 = 1e-5

# exp(x) -> exp(x.clamp(max=CUTOFF))
CUTOFF = 20
CUTOFF_eta = 5  # ttbar dataset has -5 < eta < 5


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, input_dim=1, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.weights = nn.Parameter(
            scale * torch.randn(input_dim, embed_dim // 2), requires_grad=False
        )

    def forward(self, t):
        projection = 2 * math.pi * torch.matmul(t, self.weights)
        embedding = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        return embedding


def get_eps(ref, eps=None):
    if eps is None:
        eps = torch.finfo(ref.dtype).eps
    return eps


def fourmomenta_to_jetmomenta(fourmomenta):
    pt = get_pt(fourmomenta)
    phi = get_phi(fourmomenta)
    eta = get_eta(fourmomenta)
    mass = get_mass(fourmomenta)

    jetmomenta = torch.stack((pt, phi, eta, mass), dim=-1)
    assert torch.isfinite(jetmomenta).all()
    return jetmomenta


def jetmomenta_to_fourmomenta(jetmomenta):
    pt, phi, eta, mass = torch.unbind(jetmomenta, dim=-1)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta.clamp(min=-CUTOFF, max=CUTOFF))
    E = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)

    fourmomenta = torch.stack((E, px, py, pz), dim=-1)
    assert torch.isfinite(fourmomenta).all()
    return fourmomenta


def get_pt(fourmomenta):
    return torch.sqrt(fourmomenta[..., 1] ** 2 + fourmomenta[..., 2] ** 2)


def get_phi(fourmomenta):
    return torch.arctan2(fourmomenta[..., 2], fourmomenta[..., 1])


def get_eta(fourmomenta):
    p_abs = torch.sqrt(torch.sum(fourmomenta[..., 1:] ** 2, dim=-1))
    eta = manual_eta(fourmomenta[..., 3], p_abs)
    return eta


def manual_eta(pz, pabs, eps=None):
    # stable implementation of arctanh(pz/pabs)
    if eps is None:
        eps = torch.finfo(pz.dtype).eps
    x = pz / (pabs + eps)
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    out = 0.5 * (torch.log1p(x) - torch.log1p(-x))
    return out


def get_mass(fourmomenta):
    m2 = fourmomenta[..., 0] ** 2 - torch.sum(fourmomenta[..., 1:] ** 2, dim=-1)
    m2 = m2.abs()
    m = m2.sqrt()
    return m


def ensure_angle(phi):
    return (phi + math.pi) % (2 * math.pi) - math.pi


def ensure_onshell(fourmomenta, onshell_list, onshell_mass, mass_reg=1e-1):
    onshell_mass = torch.tensor(onshell_mass, device=fourmomenta.device, dtype=fourmomenta.dtype)
    onshell_mass = onshell_mass.unsqueeze(0).expand(fourmomenta.shape[0], onshell_mass.shape[-1])
    fourmomenta[..., onshell_list, 0] = torch.sqrt(
        onshell_mass**2 + torch.sum(fourmomenta[..., onshell_list, 1:] ** 2, dim=-1)
    )

    # ensure minimal mass
    mask = get_mass(fourmomenta) < mass_reg
    fourmomenta[mask][..., 0] = ((fourmomenta[mask][..., 1:] ** 2).sum(dim=-1) + mass_reg**2).sqrt()
    return fourmomenta


def delta_phi(jetmomenta, idx1, idx2, abs=False):
    dphi = jetmomenta[..., idx1, 1] - jetmomenta[..., idx2, 1]
    dphi = ensure_angle(dphi)
    return torch.abs(dphi) if abs else dphi


def delta_eta(jetmomenta, idx1, idx2, abs=False):
    deta = jetmomenta[..., idx1, 2] - jetmomenta[..., idx2, 2]
    return torch.abs(deta) if abs else deta


def delta_r(jetmomenta, idx1, idx2):
    return (delta_phi(jetmomenta, idx1, idx2) ** 2 + delta_eta(jetmomenta, idx1, idx2) ** 2) ** 0.5


def delta_r_fast(jetmomenta1, jetmomenta2):
    dphi = ensure_angle(jetmomenta1[..., 1] - jetmomenta2[..., 1])
    deta = jetmomenta1[..., 2] - jetmomenta2[..., 2]
    return (dphi**2 + deta**2) ** 0.5


def get_virtual_particle(jetmomenta, components):
    fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)

    particle = fourmomenta[..., components, :].sum(dim=-2)
    particle = fourmomenta_to_jetmomenta(particle)
    return particle
