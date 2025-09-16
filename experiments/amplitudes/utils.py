import os
from numpy import load
import torch

from experiments.amplitudes.constants import get_mass

from lloca.utils.lorentz import lorentz_eye
from lloca.utils.rand_transforms import rand_lorentz
from lloca.utils.polar_decomposition import restframe_boost


def standardize_momentum(momentum, mean=None, std=None):
    # use common mean() and std() for all components in E, px, py, pz
    # note: empirically this step is not super important; rescaling by momentum.std() does the job
    if mean is None or std is None:
        mean = momentum.mean()
        std = momentum.std().clamp(min=1e-2)

    momentum_prepd = (momentum - mean) / std
    return momentum_prepd, mean, std


def preprocess_amplitude(amplitude, std=None, mean=None):
    log_amplitude = amplitude.log()
    if std is None or mean is None:
        mean = log_amplitude.mean()
        std = log_amplitude.std()
    prepd_amplitude = (log_amplitude - mean) / std
    return prepd_amplitude, mean, std


def undo_preprocess_amplitude(prepd_amplitude, mean, std):
    assert mean is not None and std is not None
    log_amplitude = prepd_amplitude * std + mean
    amplitude = log_amplitude.clamp(max=10).exp()
    return amplitude, log_amplitude


def load_file(
    data_path,
    cfg_data,
    dataset,
    amp_mean=None,
    amp_std=None,
    mom_std=None,
    network_float64=False,
    momentum_float64=True,
    generator=None,
):
    """
    Parameters
    ----------
    data_path : str
        Path to the data file.
    cfg_data : object
        Configuration object with data loading parameters.
    dataset : str
        Name of the dataset file, like 'zgg', 'zgggg' or 'zgggg_0'.
    amp_mean : float
        Mean of the amplitude for standardization.
    amp_std : float
        Standard deviation of the amplitude for standardization.
    mom_std : float
        Standard deviation of the momentum for standardization.
    network_float64 : bool
        Should network inputs have dtype float64? Defaults to False.
    momentum_float64 : bool
        Should momenta be stored in float64? Defaults to True.
    generator : torch.Generator
        Random generator for reproducibility. Used for AmplitudeXL loading.
    """
    network_dtype = torch.float64 if network_float64 else torch.float32
    momentum_dtype = torch.float64 if momentum_float64 else torch.float32

    assert os.path.exists(data_path), f"Data file {data_path} does not exist."
    data_raw = load(data_path)
    data_raw = torch.tensor(data_raw, dtype=momentum_dtype)

    if cfg_data.subsample is not None:
        assert cfg_data.subsample <= data_raw.shape[0]
        data_raw = data_raw[: cfg_data.subsample]

    momentum = data_raw[:, :-1]
    momentum = momentum.reshape(momentum.shape[0], momentum.shape[1] // 4, 4)
    amplitude = data_raw[:, [-1]]

    # mass regulator
    if cfg_data.mass_reg is not None:
        mass = get_mass(dataset, cfg_data.mass_reg)
        mass = torch.tensor(mass, dtype=momentum_dtype).unsqueeze(0)
        momentum[..., 0] = torch.sqrt((momentum[..., 1:] ** 2).sum(dim=-1) + mass**2)

    # prepare momenta
    if cfg_data.prepare == "lorentz":
        # boost to the center-of-mass ref. frame of incoming particles
        # then apply general Lorentz trafo L=R*B
        lab_momentum = momentum[..., :2, :].sum(dim=-2)
        to_com = restframe_boost(lab_momentum)
        trafo = rand_lorentz(
            momentum.shape[:-2], generator=generator, dtype=momentum_dtype
        )
        trafo = torch.einsum("...ij,...jk->...ik", trafo, to_com)
    elif cfg_data.prepare == "identity":
        # keep the data unchanged
        trafo = lorentz_eye(
            momentum.shape[:-2], device=momentum.device, dtype=momentum_dtype
        )
    else:
        raise ValueError(f"cfg.data.prepare={cfg_data.prepare} not implemented")
    momentum = torch.einsum("...ij,...kj->...ki", trafo, momentum)

    if mom_std is None:
        mom_std = momentum.std()
    momentum /= mom_std

    amplitude, amp_mean, amp_std = preprocess_amplitude(
        amplitude, std=amp_std, mean=amp_mean
    )

    # move everything except momentum to less safe dtype
    amplitude = amplitude.to(network_dtype)
    amp_mean = amp_mean.to(network_dtype)
    amp_std = amp_std.to(network_dtype)
    mom_std = mom_std.to(network_dtype)
    return amplitude, momentum, amp_mean, amp_std, mom_std
