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
        ortho_kwargs: dict
            Keyword arguments for orthogonalization
        """
        super().__init__()
        self.ortho_kwargs = ortho_kwargs
        self.equivectors = equivectors(n_vectors=n_vectors)
        self.is_global = is_global
        self.random = random
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
        **kwargs,
    ):
        super().__init__(*args, n_vectors=3, **kwargs)
        self.gamma_max = gamma_max
        self.gamma_hardness = gamma_hardness
        self.deterministic_boost = deterministic_boost

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
        rotation_references = [vecs[..., i, :] for i in range(1, vecs.shape[-2])]
        boost = self._deterministic_boost(boost, ptr)
        boost, reg_gammamax, gamma_mean, gamma_max = self._clamp_boost(boost)

        trafo, reg_collinear = polar_decomposition(
            boost,
            rotation_references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        tracker = {
            "reg_collinear": reg_collinear,
            "gamma_mean": gamma_mean,
            "gamma_max": gamma_max,
        }
        if reg_gammamax is not None:
            tracker["reg_gammamax"] = reg_gammamax
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames

    def _clamp_boost(self, x):
        mass = lorentz_squarednorm(x).clamp(min=0).sqrt().unsqueeze(-1)
        beta = x[..., 1:] / x[..., [0]].clamp(min=1e-10)
        gamma = x[..., [0]] / mass
        gamma_max = gamma.max().detach().cpu()
        gamma_mean = gamma.detach().mean().cpu()

        if self.gamma_max is None:
            return x, None, gamma_mean, gamma_max

        else:
            # carefully clamp gamma to keep boosts under control
            reg_gammamax = (gamma > self.gamma_max).sum().cpu()
            gamma_reg = soft_clamp(
                gamma, min=1, max=self.gamma_max, hardness=self.gamma_hardness
            )
            beta_scaling = (
                torch.sqrt(
                    torch.clamp(1 - 1 / gamma_reg.clamp(min=1e-10).square(), min=1e-10)
                )
                / (beta**2).sum(dim=-1, keepdim=True).clamp(min=1e-10).sqrt()
            )
            beta_reg = beta * beta_scaling
            x_reg = mass * torch.cat((gamma_reg, gamma_reg * beta_reg), dim=-1)
            return x_reg, reg_gammamax, gamma_mean, gamma_max

    def _deterministic_boost(self, boost, ptr):
        if self.deterministic_boost is None:
            pass
        elif self.deterministic_boost == "global":
            # average boost vector over the event
            boost = average_event(boost, ptr)
        elif self.deterministic_boost == "local":
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
                f"Option deterministic_boost={self.deterministic_boost} not implemented"
            )

        return boost


class LearnedSO13Frames(LearnedFrames):
    """Frames as orthonormal set of Lorentz vectors."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, n_vectors=3, **kwargs)

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
        vecs = [vecs[..., i, :] for i in range(vecs.shape[-2])]

        trafo, reg_lightlike, reg_coplanar = orthogonalize_4d(
            vecs, **self.ortho_kwargs, return_reg=True
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
        **kwargs,
    ):
        super().__init__(*args, n_vectors=2, **kwargs)

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
        references = [references[..., i, :] for i in range(references.shape[-2])]

        trafo, reg_collinear = polar_decomposition(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        tracker = {"reg_collinear": reg_collinear}
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


class LearnedSO3Frames(LearnedFrames):
    """Frames from SO(3) rotations.

    This is a special case of LearnedPolarDecompositionFrames
    where the first vector is trivial (1,0,0,0)."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.n_vectors = 2
        super().__init__(
            *args,
            n_vectors=self.n_vectors,
            **kwargs,
        )

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
        references = [references[..., i, :] for i in range(self.n_vectors)]

        trafo, reg_collinear = polar_decomposition(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
        )
        tracker = {"reg_collinear": reg_collinear}
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


class LearnedSO2Frames(LearnedFrames):
    """Frames from SO(2) rotations around the beam axis.

    This is a special case of LearnedPolarDecompositionFrames
    where the firsts two vectors are trivial (1,0,0,0) and (0,0,0,1)."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.n_vectors = 1
        super().__init__(
            *args,
            n_vectors=self.n_vectors,
            **kwargs,
        )

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
        references = [
            lorentz_eye(
                fourmomenta.shape[:-1],
                device=fourmomenta.device,
                dtype=fourmomenta.dtype,
            )[..., 3]
        ]  # difference 2 compared LearnedPolarDecompositionFrames
        references.append(extra_references[..., 0, :])

        trafo, reg_collinear = polar_decomposition(
            fourmomenta,
            references,
            **self.ortho_kwargs,
            return_reg=True,
        )

        tracker = {"reg_collinear": reg_collinear}
        frames = Frames(trafo, is_global=self.is_global)
        return (frames, tracker) if return_tracker else frames


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
