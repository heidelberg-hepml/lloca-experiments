import torch

EPS = 1e-10
CUTOFF = 10

# weaver defaults for tagging features standardization (mean, std)
TAGGING_FEATURES_PREPROCESSING = [
    [1.7, 0.7],  # log_pt
    [2.0, 0.7],  # log_energy
    [-4.7, 0.7],  # log_pt_rel
    [-4.7, 0.7],  # log_energy_rel
    [0, 1],  # dphi
    [0, 1],  # deta
    [0.2, 4],  # dr
]


def get_pt(p):
    # transverse momentum
    return torch.sqrt((p[..., 1] ** 2 + p[..., 2] ** 2).clamp(min=EPS))


def stable_arctanh(x, eps=1e-10):
    # implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def avoid_zero(x, eps=EPS):
    # set small-abs values to eps for numerical stability
    return torch.where(x.abs() < eps, eps, x)


def get_phi(p):
    # azimuthal angle
    return torch.arctan2(avoid_zero(p[..., 2]), avoid_zero(p[..., 1]))


def get_eta(p):
    # rapidity
    p_abs = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1).clamp(min=EPS))
    return stable_arctanh(p[..., 3] / p_abs)


def get_tagging_features(fourmomenta, jet, eps=1e-10):
    """
    Compute features typically used in jet tagging

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (n_particles, 4)
        Fourmomenta in the format (E, px, py, pz)
    jet: torch.tensor of shape (n_particles, 4)
        Jet momenta in the shape (E, px, py, pz)
    eps: float

    Returns
    -------
    features: torch.tensor of shape (n_particles, 7)
        Features: log_pt, log_energy, log_pt_rel, log_energy_rel, dphi, deta, dr
    """
    log_pt = get_pt(fourmomenta).unsqueeze(-1).log()
    log_energy = fourmomenta[..., 0].unsqueeze(-1).clamp(min=eps).log()

    log_pt_rel = (get_pt(fourmomenta).log() - get_pt(jet).log()).unsqueeze(-1)
    log_energy_rel = (
        fourmomenta[..., 0].clamp(min=eps).log() - jet[..., 0].clamp(min=eps).log()
    ).unsqueeze(-1)
    phi_4, phi_jet = get_phi(fourmomenta), get_phi(jet)
    dphi = ((phi_4 - phi_jet + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    eta_4, eta_jet = get_eta(fourmomenta), get_eta(jet)
    deta = -(eta_4 - eta_jet).unsqueeze(-1)
    dr = torch.sqrt((dphi**2 + deta**2).clamp(min=eps))
    features = [
        log_pt,
        log_energy,
        log_pt_rel,
        log_energy_rel,
        dphi,
        deta,
        dr,
    ]
    for i, feature in enumerate(features):
        mean, factor = TAGGING_FEATURES_PREPROCESSING[i]
        features[i] = (feature - mean) * factor
    features = torch.cat(features, dim=-1)
    return features
