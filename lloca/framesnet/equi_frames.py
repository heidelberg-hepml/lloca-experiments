"""Equivariant local frames for various symmetry groups."""
import torch
from torch_geometric.utils import scatter

from .frames import Frames
from .nonequi_frames import FramesPredictor
from ..utils.utils import get_batch_from_ptr
from ..utils.polar_decomposition import polar_decomposition
from ..utils.lorentz import lorentz_eye, lorentz_squarednorm
from ..utils.orthogonalize_4d import orthogonalize_4d


class LearnedFrames(FramesPredictor):
    """Abstract class for local Frames constructed
    based on equivariantly predicted vectors"""

    def __init__(
        self,
        equivectors,
        n_vectors,
        is_global=False,
        random=False,
        fix_params=False,
        checks=False,
        ortho_kwargs={},
    ):
        """
        Parameters
        ----------
        equivectors: nn.Module
            Network that equivariantly predicts vectors
        n_vectors: int
            Number of vectors to predict
        is_global: bool
            If True, average the predicted vectors to construct a global frame
        random: bool
            If True, re-initialize the equivectors at each forward pass
            This is a fancy way of doing data augmentation
        fix_params: bool
            Like random, but without the resampling
        checks : bool
            If True, perform additional assertion checks on predicted vectors
        ortho_kwargs: dict
            Keyword arguments for orthogonalization
        """
        super().__init__()
        self.ortho_kwargs = ortho_kwargs
        self.equivectors = equivectors(n_vectors=n_vectors)
        self.is_global = is_global
        self.random = random
        self.checks = checks
        if random or fix_params:
            self.equivectors.requires_grad_(False)

    def init_weights_or_not(self):
        if self.random and self.training:
            self.equivectors.apply(init_weights)

    def globalize_vecs_or_not(self, vecs, ptr):
        return average_event(vecs, ptr) if self.is_global else vecs

    def __repr__(self):
        classname = self.__class__.__name__
        method = self.ortho_kwargs["method"]
        string = f"{classname}(method={method})"
        return string


class LearnedPDFrames(LearnedFrames):
    """Frames as learnable polar decompositions.

    This is our default approach.
    LearnedSO13Frames works similarly well, but is less flexible.
    """

    def __init__(
        self,
        *args,
        gamma_max=None,
        gamma_hardness=None,
        deterministic_boost=None,
        compile=False,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=3, **kwargs)
        self.gamma_max = gamma_max
        self.gamma_hardness = gamma_hardness
        self.deterministic_boost = deterministic_boost

        if compile:
            self.polar_decomposition = torch.compile(
                polar_decomposition, dynamic=True, fullgraph=True
            )
        else:
            self.polar_decomposition = polar_decomposition

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        """
        Parameters
        ----------
        fourmomenta: torch.Tensor
            Tensor of shape (..., 4) containing the four-momenta
        scalars: torch.Tensor or None
            Optional tensor of shape (..., n_scalars) containing additional scalar features
        ptr: torch.Tensor or None
            Pointer for sparse tensors, or None for dense tensors
        return_tracker: bool
            If True, return a tracker dictionary with regularization information

        Returns
        -------
        Frames
            Local frames constructed from the polar decomposition of the four-momenta
        tracker: dict (optional)
            Dictionary containing regularization information, if return_tracker is True
        """
        self.init_weights_or_not()
        vecs = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        vecs = self.globalize_vecs_or_not(vecs, ptr)
        boost = vecs[..., 0, :]
        rotation_references = vecs[..., 1:, :]
        boost = deterministic_boost(
            boost, ptr, deterministic_boost=self.deterministic_boost
        )
        boost, reg_gammamax, gamma_mean, gamma_max = clamp_boost(
            boost, gamma_max=self.gamma_max, gamma_hardness=self.gamma_hardness
        )

        trafo, reg_lightlike, reg_collinear = self.polar_decomposition(
            boost,
            rotation_references,
            **self.ortho_kwargs,
            return_reg=True,
            checks=self.checks,
        )
        tracker = {
            "reg_lightlike": reg_lightlike,
            "reg_collinear": reg_collinear,
            "gamma_mean": gamma_mean,
            "gamma_max": gamma_max,
        }
        if reg_gammamax is not None:
            tracker["reg_gammamax"] = reg_gammamax
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


class LearnedSO13Frames(LearnedFrames):
    """Frames as orthonormal set of Lorentz vectors."""

    def __init__(
        self,
        *args,
        compile=False,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=3, **kwargs)
        if compile:
            self.orthogonalize_4d = torch.compile(
                orthogonalize_4d, dynamic=True, fullgraph=True
            )
        else:
            self.orthogonalize_4d = orthogonalize_4d

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        """
        Parameters
        ----------
        fourmomenta: torch.Tensor
            Tensor of shape (..., 4) containing the four-momenta
        scalars: torch.Tensor or None
            Optional tensor of shape (..., n_scalars) containing additional scalar features
        ptr: torch.Tensor or None
            Pointer for sparse tensors, or None for dense tensors
        return_tracker: bool
            If True, return a tracker dictionary with regularization information

        Returns
        -------
        Frames
            Local frames constructed from the polar decomposition of the four-momenta
        tracker: dict (optional)
            Dictionary containing regularization information, if return_tracker is True
        """
        self.init_weights_or_not()
        vecs = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        vecs = self.globalize_vecs_or_not(vecs, ptr)

        trafo, reg_lightlike, reg_coplanar = self.orthogonalize_4d(
            vecs, **self.ortho_kwargs, return_reg=True, checks=self.checks
        )

        tracker = {"reg_lightlike": reg_lightlike, "reg_coplanar": reg_coplanar}
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


class LearnedRestFrames(LearnedFrames):
    """Rest frame transformation with learnable rotation.

    This is a special case of LearnedPolarDecompositionFrames
    where the boost vector is chosen to be the particle momentum.
    Note that the rotation is constructed equivariantly to get
    the correct transformation behaviour of local frames.
    """

    def __init__(
        self,
        *args,
        compile=False,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=2, **kwargs)
        if compile:
            self.polar_decomposition = torch.compile(
                polar_decomposition, dynamic=True, fullgraph=True
            )
        else:
            self.polar_decomposition = polar_decomposition

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        """
        Parameters
        ----------
        fourmomenta: torch.Tensor
            Tensor of shape (..., 4) containing the four-momenta
        scalars: torch.Tensor or None
            Optional tensor of shape (..., n_scalars) containing additional scalar features
        ptr: torch.Tensor or None
            Pointer for sparse tensors, or None for dense tensors
        return_tracker: bool
            If True, return a tracker dictionary with regularization information

        Returns
        -------
        Frames
            Local frames constructed from the polar decomposition of the four-momenta
        tracker: dict (optional)
            Dictionary containing regularization information, if return_tracker is True
        """
        self.init_weights_or_not()
        references = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        references = self.globalize_vecs_or_not(references, ptr)

        trafo, reg_lightlike, reg_collinear = self.polar_decomposition(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
            checks=self.checks,
        )
        tracker = {"reg_lightlike": reg_lightlike, "reg_collinear": reg_collinear}
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


class LearnedSO3Frames(LearnedFrames):
    """Frames from SO(3) rotations.

    This is a special case of LearnedPolarDecompositionFrames
    where the first vector is trivial (1,0,0,0)."""

    def __init__(
        self,
        *args,
        compile=False,
        **kwargs,
    ):
        self.n_vectors = 2
        super().__init__(
            *args,
            n_vectors=self.n_vectors,
            **kwargs,
        )
        if compile:
            self.polar_decomposition = torch.compile(
                polar_decomposition, dynamic=True, fullgraph=True
            )
        else:
            self.polar_decomposition = polar_decomposition

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        """
        Parameters
        ----------
        fourmomenta: torch.Tensor
            Tensor of shape (..., 4) containing the four-momenta
        scalars: torch.Tensor or None
            Optional tensor of shape (..., n_scalars) containing additional scalar features
        ptr: torch.Tensor or None
            Pointer for sparse tensors, or None for dense tensors
        return_tracker: bool
            If True, return a tracker dictionary with regularization information

        Returns
        -------
        Frames
            Local frames constructed from the polar decomposition of the four-momenta
        tracker: dict (optional)
            Dictionary containing regularization information, if return_tracker is True
        """
        self.init_weights_or_not()
        references = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        references = self.globalize_vecs_or_not(references, ptr)
        fourmomenta = lorentz_eye(
            fourmomenta.shape[:-1], device=fourmomenta.device, dtype=fourmomenta.dtype
        )[
            ..., 0
        ]  # only difference compared to LearnedPolarDecompositionFrames

        trafo, reg_lightlike, reg_collinear = self.polar_decomposition(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
            checks=self.checks,
        )
        tracker = {"reg_lightlike": reg_lightlike, "reg_collinear": reg_collinear}
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


class LearnedZFrames(LearnedFrames):
    """Frames from ztransform,
    i.e. combination of boost along z and rotation around z axis,
    or SO(1,1)_z x SO(2)_z.

    This is a special case of LearnedPolarDecompositionFrames
    where the boost vector is constrained to point along the z-axis
    and one of the rotation references is (0,0,0,1)."""

    def __init__(
        self,
        *args,
        gamma_max=None,
        gamma_hardness=None,
        compile=False,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=2, **kwargs)
        self.gamma_max = gamma_max
        self.gamma_hardness = gamma_hardness
        if compile:
            self.polar_decomposition = torch.compile(
                polar_decomposition, dynamic=True, fullgraph=True
            )
        else:
            self.polar_decomposition = polar_decomposition

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        """
        Parameters
        ----------
        fourmomenta: torch.Tensor
            Tensor of shape (..., 4) containing the four-momenta
        scalars: torch.Tensor or None
            Optional tensor of shape (..., n_scalars) containing additional scalar features
        ptr: torch.Tensor or None
            Pointer for sparse tensors, or None for dense tensors
        return_tracker: bool
            If True, return a tracker dictionary with regularization information

        Returns
        -------
        Frames
            Local frames constructed from the polar decomposition of the four-momenta
        tracker: dict (optional)
            Dictionary containing regularization information, if return_tracker is True
        """
        self.init_weights_or_not()
        vecs = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        vecs = self.globalize_vecs_or_not(vecs, ptr)
        boost = vecs[..., 0, :]
        boost[..., [1, 2]] = 0.0  # only z-boost (keeps timelike vectors timelike)
        rotation_references = vecs[..., 1, :]
        boost, reg_gammamax, gamma_mean, gamma_max = clamp_boost(
            boost, gamma_max=self.gamma_max, gamma_hardness=self.gamma_hardness
        )

        spurion_references = lorentz_eye(
            fourmomenta.shape[:-1],
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )[
            ..., 3
        ]  # difference 2 compared LearnedPolarDecompositionFrames
        rotation_references = torch.stack(
            [rotation_references, spurion_references], dim=-2
        )

        trafo, reg_lightlike, reg_collinear = self.polar_decomposition(
            boost,
            rotation_references,
            **self.ortho_kwargs,
            return_reg=True,
            checks=self.checks,
        )
        tracker = {
            "reg_lightlike": reg_lightlike,
            "reg_collinear": reg_collinear,
            "gamma_mean": gamma_mean,
            "gamma_max": gamma_max,
        }
        if reg_gammamax is not None:
            tracker["reg_gammamax"] = reg_gammamax
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


class LearnedSO2Frames(LearnedFrames):
    """Frames from SO(2) rotations around the beam axis.

    This is a special case of LearnedPolarDecompositionFrames
    where the firsts two vectors are trivial (1,0,0,0) and (0,0,0,1)."""

    def __init__(
        self,
        *args,
        compile=False,
        **kwargs,
    ):
        self.n_vectors = 1
        super().__init__(
            *args,
            n_vectors=self.n_vectors,
            **kwargs,
        )
        if compile:
            self.polar_decomposition = torch.compile(
                polar_decomposition, dynamic=True, fullgraph=True
            )
        else:
            self.polar_decomposition = polar_decomposition

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        """
        Parameters
        ----------
        fourmomenta: torch.Tensor
            Tensor of shape (..., 4) containing the four-momenta
        scalars: torch.Tensor or None
            Optional tensor of shape (..., n_scalars) containing additional scalar features
        ptr: torch.Tensor or None
            Pointer for sparse tensors, or None for dense tensors
        return_tracker: bool
            If True, return a tracker dictionary with regularization information

        Returns
        -------
        Frames
            Local frames constructed from the polar decomposition of the four-momenta
        tracker: dict (optional)
            Dictionary containing regularization information, if return_tracker is True
        """
        self.init_weights_or_not()
        references = self.equivectors(fourmomenta, scalars=scalars, ptr=ptr)
        extra_references = self.globalize_vecs_or_not(references, ptr)
        fourmomenta = lorentz_eye(
            fourmomenta.shape[:-1], device=fourmomenta.device, dtype=fourmomenta.dtype
        )[
            ..., 0
        ]  # difference 1 compared LearnedPolarDecompositionFrames
        spurion_references = lorentz_eye(
            fourmomenta.shape[:-1],
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )[
            ..., 3
        ]  # difference 2 compared LearnedPolarDecompositionFrames
        references = torch.stack(
            [spurion_references, extra_references[..., 0, :]], dim=-2
        )

        trafo, reg_lightlike, reg_collinear = self.polar_decomposition(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
            checks=self.checks,
        )

        tracker = {"reg_lightlike": reg_lightlike, "reg_collinear": reg_collinear}
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


def clamp_boost(x, gamma_max, gamma_hardness):
    mass = lorentz_squarednorm(x).clamp(min=0).sqrt().unsqueeze(-1)
    t0 = x.narrow(-1, 0, 1)
    beta = x[..., 1:] / t0.clamp_min(1e-10)
    gamma = t0 / mass
    gamma_max_realized = gamma.max().detach()
    gamma_mean = gamma.mean().detach()

    if gamma_max is None:
        return x, None, gamma_mean, gamma_max_realized

    else:
        # carefully clamp gamma to keep boosts under control
        reg_gammamax = (gamma > gamma_max).sum().detach()
        gamma_reg = soft_clamp(gamma, min=1, max=gamma_max, hardness=gamma_hardness)
        beta_scaling = (
            torch.sqrt(
                torch.clamp(1 - 1 / gamma_reg.clamp(min=1e-10).square(), min=1e-10)
            )
            / (beta**2).sum(dim=-1, keepdim=True).clamp(min=1e-10).sqrt()
        )
        beta_reg = beta * beta_scaling
        x_reg = mass * torch.cat((gamma_reg, gamma_reg * beta_reg), dim=-1)
        return x_reg, reg_gammamax, gamma_mean, gamma_max_realized


def deterministic_boost(boost, ptr, deterministic_boost):
    if deterministic_boost is None:
        pass
    elif deterministic_boost == "global":
        # average boost vector over the event
        boost = average_event(boost, ptr)
    elif deterministic_boost == "local":
        # average boost over all other particles in the event
        boost_averaged = average_event(boost, ptr)
        if ptr is None:
            nparticles = boost.shape[1]
        else:
            diff = ptr[1:] - ptr[:-1]
            nparticles = (diff).repeat_interleave(diff).unsqueeze(-1)
        boost = boost_averaged - boost / nparticles
    else:
        raise ValueError(
            f"Option deterministic_boost={deterministic_boost} not implemented"
        )

    return boost


def average_event(vecs, ptr=None):
    """Average vectors across events and expand again.

    Parameters
    ----------
    vecs: torch.Tensor
        Tensor of shape (..., n_vectors, 4)
        where the last dimension contains the vectors
    ptr: torch.Tensor or None
        Pointer to the batch of events, or None for global averaging

    Returns
    -------
    torch.Tensor
        Averaged vectors of shape (..., n_vectors, 4).
    """
    if ptr is None:
        vecs = vecs.mean(dim=1, keepdim=True).expand_as(vecs)
    else:
        batch = get_batch_from_ptr(ptr)
        vecs = scatter(vecs, batch, dim=0, reduce="mean").index_select(0, batch)
    return vecs


def init_weights(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def soft_clamp(x, max=None, min=None, hardness=None):
    if hardness is None:
        # hard clamp
        return x.clamp(min=min, max=max)
    else:
        # soft clamp (better gradients)
        out = max - torch.nn.functional.softplus(max - x, beta=hardness)
        return out.clamp(min=min)
