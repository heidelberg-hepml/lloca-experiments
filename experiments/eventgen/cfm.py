import torch
from torch import nn
from torch.autograd import grad
from torchdiffeq import odeint

import experiments.eventgen.coordinates as c
from experiments.eventgen.distributions import (
    BaseDistribution,
    NaivePPPLogM2,
    NaivePPPM2,
    StandardLogPtPhiEtaLogM2,
    StandardPPPLogM2,
)
from experiments.eventgen.geometry import BaseGeometry, SimplePossiblyPeriodicGeometry
from experiments.eventgen.utils import GaussianFourierProjection
from experiments.logger import LOGGER


def hutchinson_trace(x_out, x_in):
    # Hutchinson's trace Jacobian estimator, needs O(1) calls to autograd
    noise = torch.randint_like(x_in, low=0, high=2).float() * 2 - 1.0
    x_out_noise = torch.sum(x_out * noise)
    gradient = grad(x_out_noise, x_in)[0].detach()
    return torch.sum(gradient * noise, dim=[1, 2])


def autograd_trace(x_out, x_in):
    # Standard way of calculating trace of the Jacobian, needs O(n) calls to autograd
    trJ = 0.0
    for i in range(x_out.shape[1]):
        for j in range(x_out.shape[2]):
            trJ += (
                grad(x_out[:, i, j].sum(), x_in, retain_graph=True)[0]
                .contiguous()[:, i, j]
                .contiguous()
                .detach()
            )
    return trJ.contiguous()


class CFM(nn.Module):
    """
    Base class for all CFM models
    - event-generation-specific features are implemented in EventCFM
    - get_velocity is implemented by architecture-specific subclasses
    """

    def __init__(
        self,
        cfm,
        odeint={"method": "dopri5", "atol": 1e-5, "rtol": 1e-5, "options": None},
        network_float64=False,
        momentum_float64=True,
    ):
        super().__init__()
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=cfm.embed_t_dim, scale=cfm.embed_t_scale),
            nn.Linear(cfm.embed_t_dim, cfm.embed_t_dim),
        )
        self.trace_fn = hutchinson_trace if cfm.hutchinson else autograd_trace
        self.odeint = odeint
        self.cfm = cfm
        self.network_dtype = torch.float64 if network_float64 else torch.float32
        self.momentum_dtype = torch.float64 if momentum_float64 else torch.float32

        # initialize to base objects, this will be overwritten later
        self.distribution = BaseDistribution()
        self.coordinates = c.BaseCoordinates()
        self.geometry = BaseGeometry()

    def init_distribution(self):
        raise NotImplementedError

    def init_coordinates(self):
        raise NotImplementedError

    def init_geometry(self):
        raise NotImplementedError

    def sample_base(self, shape, device, dtype, generator=None):
        fourmomenta = self.distribution.sample(shape, device, dtype, generator=generator)
        return fourmomenta

    def get_velocity(self, x, t):
        """
        Parameters
        ----------
        x : torch.tensor with shape (batchsize, n_particles, 4)
        t : torch.tensor with shape (batchsize, 1, 1)
        """
        # implemented by architecture-specific subclasses
        raise NotImplementedError

    def handle_velocity(self, v):
        # default: do nothing
        return v

    def batch_loss(self, fm0):
        """
        Construct the conditional flow matching objective.
        Note: t=0/t=1 are target/base distribution.

        Parameters
        ----------
        fm0 : torch.tensor with shape (batchsize, n_particles, 4)
            Target space particles in fourmomenta space

        Returns
        -------
        loss : torch.tensor with shape (1)
        """
        t = torch.rand(
            fm0.shape[0],
            1,
            1,
            dtype=self.network_dtype,
            device=fm0.device,
        )
        fm1 = self.sample_base(fm0.shape, fm0.device, fm0.dtype)

        # construct target trajectories
        x0 = self.coordinates.fourmomenta_to_x(fm0)
        x1 = self.coordinates.fourmomenta_to_x(fm1)
        xt, vt_x = self.geometry.get_trajectory(x0, x1, t)
        vp_x, tracker = self.get_velocity(xt, t)

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(vp_x, vt_x).mean()
        for k in range(4):
            tracker[f"mse_{k}"] = ((vp_x - vt_x) ** 2)[..., k].mean().detach().cpu()
        return distance, tracker

    def sample(self, shape, device):
        """
        Sample from CFM model
        Solve an ODE using a NN-parametrized velocity field

        Parameters
        ----------
        shape : List[int]
            Shape of events that should be generated
        device : torch.device

        Returns
        -------
        fm0 : torch.tensor with shape shape = (batchsize, n_particles, 4)
            Generated events
        """

        def velocity(t, xt):
            xt = self.geometry._handle_periodic(xt)
            t = t * torch.ones(shape[0], 1, 1, dtype=self.network_dtype, device=xt.device)
            vp_x = self.get_velocity(xt, t)[0]
            vp_x = self.handle_velocity(vp_x)
            return vp_x

        # sample fourmomenta from base distribution
        fm1 = self.sample_base(shape, device, dtype=self.momentum_dtype)
        x1 = self.coordinates.fourmomenta_to_x(fm1)

        # solve ODE in straight space
        x0 = odeint(
            velocity,
            x1,
            torch.tensor([1.0, 0.0], dtype=self.network_dtype, device=x1.device),
            **self.odeint,
        )[-1]

        # the infamous nan remover
        # (MLP sometimes returns nan for single events,
        # and all components of the event are nan...
        # just sample another event in this case)
        mask = torch.isfinite(x0).all(dim=[1, 2])
        if (~mask).any():
            mask2 = torch.isfinite(x0)
            x0 = x0[mask, ...]
            fm1 = fm1[mask, ...]
            LOGGER.warning(f"Found {(~mask2).sum(dim=0)} nan events while sampling")

        # transform generated event back to fourmomenta
        fm0 = self.coordinates.x_to_fourmomenta(x0)
        return fm0

    def log_prob(self, fm0):
        """
        Evaluate log_prob for existing target samples in a CFM model
        Solve ODE involving the trace of the velocity field, this is more expensive than normal sampling
        The 'self.hutchinson' parameter controls if the trace should be evaluated
        with the hutchinson trace estimator that needs O(1) calls to the network,
        as opposed to the exact autograd trace that needs O(n_particles) calls to the network
        Note: Could also have a sample_and_log_prob method, but we have no use case for this

        Parameters
        ----------
        fm0 : torch.tensor with shape (batchsize, n_particles, 4)
            Target space particles in fourmomenta space

        Returns
        -------
        log_prob_fm : torch.tensor with shape (batchsize)
            log_prob of each event in x0, evaluated in fourmomenta space
        """

        @torch.set_grad_enabled(True)
        def net_wrapper(t, state):
            xt = state[0]
            xt = self.geometry._handle_periodic(xt)
            xt = xt.detach().requires_grad_(True)
            t = t * torch.ones(
                xt.shape[0],
                1,
                1,
                dtype=self.network_dtype,
                device=xt.device,
            )
            vp_x = self.get_velocity(xt, t)[0]
            vp_x = self.handle_velocity(vp_x)
            dlogp_dt_x = -self.trace_fn(vp_x, xt).unsqueeze(-1)
            return vp_x.detach(), dlogp_dt_x.detach()

        # solve ODE in coordinates
        x0 = self.coordinates.fourmomenta_to_x(fm0)
        logdetjac0_cfm_x = torch.zeros(
            (x0.shape[0], 1),
            dtype=self.network_dtype,
            device=x0.device,
        )
        state0 = (x0, logdetjac0_cfm_x)
        xt, logdetjact_cfm_x = odeint(
            net_wrapper,
            state0,
            torch.tensor([0.0, 1.0], dtype=self.network_dtype, device=x0.device),
            **self.odeint,
        )
        logdetjac_cfm_x = logdetjact_cfm_x[-1].detach()
        x1 = xt[-1].detach()

        # the infamous nan remover
        # (MLP sometimes returns nan for single events,
        # just remove these events from the log_prob computation)
        mask = torch.isfinite(x1).all(dim=[1, 2])
        if (~mask).any():
            mask2 = torch.isfinite(x1)
            logdetjac_cfm_x = logdetjac_cfm_x[mask]
            x1 = x1[mask]
            fm0 = fm0[mask]
            LOGGER.warning(f"Found {(~mask2).sum(dim=0)} nan events in log_prob")

        fm1 = self.coordinates.x_to_fourmomenta(x1)
        logdetjac_forward = self.coordinates.logdetjac_fourmomenta_to_x(fm0)[0]
        logdetjac_inverse = -self.coordinates.logdetjac_fourmomenta_to_x(fm1)[0]

        # collect log_probs
        log_prob_base_fm = self.distribution.log_prob(fm1)
        log_prob_fm = log_prob_base_fm - logdetjac_cfm_x - logdetjac_forward - logdetjac_inverse

        mask = log_prob_fm.abs() < 100
        if (~mask).any():
            LOGGER.warning(
                f"Removing {(~mask).sum(dim=0)} events with large log_prob_fm {log_prob_fm[~mask]}. "
                f"fw {logdetjac_forward[~mask]}, inv {logdetjac_inverse[~mask]}"
            )
            log_prob_fm = log_prob_fm[mask]
        return log_prob_fm


class EventCFM(CFM):
    """
    Add event-generation-specific methods to CFM classes:
    - Save information at the wrapper level
    - Handle base distribution and coordinates for RFM
    - Wrapper-specific preprocessing and undo_preprocessing
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_physics(
        self,
        units,
        pt_min,
        delta_r_min,
        onshell_list,
        onshell_mass,
        virtual_components,
        base_type,
        use_pt_min,
        use_delta_r_min,
    ):
        """
        Pass physics information to the CFM class

        Parameters
        ----------
        units: float
            Scale of dimensionful quantities
            I call it 'units' because we can really choose it arbitrarily without losing anything
            Hard-coded in EventGenerationExperiment
        pt_min: List[float]
            Minimum pt value for each particle
            Hard-coded in EventGenerationExperiment
        delta_r_min: float
            Minimum delta_r value
            We do not support different minimum delta_r for each particle yet
            Hard-coded in EventGenerationExperiment
        onshell_list: List[int]
            Indices of the onshell particles
            Hard-coded in EventGenerationExperiment
        onshell_mass: List[float]
            Masses of the onshell particles in the same order as in onshell_list
            Hard-coded in EventGenerationExperiment
        virtual_components: List[List[int]]
            Indices of the virtual particles
        base_type: int
            Which base distribution to use
        use_delta_r_min: bool
            Whether the base distribution should have delta_r cuts
        use_pt_min: bool
            Whether the base distribution should have pt cuts
        """
        self.units = units
        self.pt_min = pt_min
        self.delta_r_min = delta_r_min
        self.onshell_list = onshell_list
        self.onshell_mass = onshell_mass
        self.virtual_components = virtual_components
        self.base_type = base_type
        self.use_delta_r_min = use_delta_r_min
        self.use_pt_min = use_pt_min

        # same preprocessing for all multiplicities
        self.prep_params = {}

    def init_distribution(self):
        args = [
            self.onshell_list,
            self.onshell_mass,
            self.units,
            self.delta_r_min,
            self.pt_min,
            self.use_delta_r_min,
            self.use_pt_min,
        ]
        if self.base_type == 1:
            self.distribution = NaivePPPM2(*args)
        elif self.base_type == 2:
            self.distribution = NaivePPPLogM2(*args)
        elif self.base_type == 3:
            self.distribution = StandardPPPLogM2(*args)
        elif self.base_type == 4:
            self.distribution = StandardLogPtPhiEtaLogM2(*args)
        else:
            raise ValueError(f"base_type={self.base_type} not implemented")

    def init_coordinates(self):
        self.coordinates = self._init_coordinates(self.cfm.coordinates)

    def _init_coordinates(self, coordinates_label):
        if coordinates_label == "Fourmomenta":
            coordinates = c.Fourmomenta()
        elif coordinates_label == "PPPM2":
            coordinates = c.PPPM2()
        elif coordinates_label == "PPPLogM2":
            coordinates = c.PPPLogM2()
        elif coordinates_label == "StandardPPPLogM2":
            coordinates = c.StandardPPPLogM2(self.onshell_list)
        elif coordinates_label == "EPhiPtPz":
            coordinates = c.EPhiPtPz()
        elif coordinates_label == "PtPhiEtaE":
            coordinates = c.PtPhiEtaE()
        elif coordinates_label == "PtPhiEtaM2":
            coordinates = c.PtPhiEtaM2()
        elif coordinates_label == "LogPtPhiEtaE":
            coordinates = c.LogPtPhiEtaE(self.pt_min, self.units)
        elif coordinates_label == "LogPtPhiEtaM2":
            coordinates = c.LogPtPhiEtaM2(self.pt_min, self.units)
        elif coordinates_label == "PtPhiEtaLogM2":
            coordinates = c.PtPhiEtaLogM2()
        elif coordinates_label == "LogPtPhiEtaLogM2":
            coordinates = c.LogPtPhiEtaLogM2(self.pt_min, self.units)
        elif coordinates_label == "StandardLogPtPhiEtaLogM2":
            coordinates = c.StandardLogPtPhiEtaLogM2(
                self.pt_min,
                self.units,
            )
        else:
            raise ValueError(f"coordinates={coordinates_label} not implemented")
        return coordinates

    def init_geometry(self):
        # placeholder for any initialization that needs to be done
        if self.cfm.geometry.type == "simple":
            self.geometry = SimplePossiblyPeriodicGeometry(
                contains_phi=self.coordinates.contains_phi,
                periodic=self.cfm.geometry.periodic,
            )
        else:
            raise ValueError(f"geometry={self.cfm.geometry} not implemented")

    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        return fourmomenta

    def undo_preprocess(self, fourmomenta):
        fourmomenta = fourmomenta * self.units
        return fourmomenta

    def sample(self, *args, **kwargs):
        fourmomenta = super().sample(*args, **kwargs)
        return fourmomenta

    def handle_velocity(self, v):
        if self.coordinates.contains_mass:
            # manually set mass velocity of onshell events to zero
            v[..., self.onshell_list, 3] = 0.0
        return v
