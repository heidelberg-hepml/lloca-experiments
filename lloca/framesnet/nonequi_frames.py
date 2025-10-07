"""Non-equivariant local frames for non-equivariant networks and data augmentation."""
import torch

from .frames import Frames
from ..utils.rand_transforms import (
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
    rand_ztransform,
)
from ..utils.polar_decomposition import restframe_boost


class FramesPredictor(torch.nn.Module):
    def __init__(self, is_global=False, is_identity=False):
        super().__init__()
        self.is_global = is_global
        self.is_identity = is_identity

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        raise NotImplementedError


class IdentityFrames(FramesPredictor):
    """Identity frames for non-equivariant networks"""

    def __init__(self):
        super().__init__(is_global=True, is_identity=True)

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        frames = Frames(
            is_identity=True,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
            shape=fourmomenta.shape[:-1],
        )

        return (frames, {}) if return_tracker else frames

    def __repr__(self):
        return "IdentityFrames()"


class RandomFrames(FramesPredictor):
    """Random frames for data augmentation."""

    def __init__(
        self,
        transform_type="lorentz",
        is_global=True,
        std_eta=0.1,
        n_max_std_eta=3.0,
    ):
        """
        Parameters
        ----------
        transform_type : str
            Type of random transformation. One of "lorentz", "rotation", "xyrotation", "ztransform".
        is_global : bool
            Global or local data augmentations, the default is global.
            Local data augmentations are a weird thing, we implemented them because we can.
        std_eta : float
            Standard deviation of the rapidity eta for the random boost.
            Only relevant if transform_type is "lorentz" or "ztransform".
        n_max_std_eta : float
            Maximum rapidity in units of std_eta.
            Only relevant if transform_type is "lorentz" or "ztransform".
        """
        super().__init__(is_global=is_global)
        self.is_global = is_global
        self.std_eta = std_eta
        self.transform_type = transform_type
        self.n_max_std_eta = n_max_std_eta

    def transform(self, shape, device, dtype):
        if self.transform_type == "lorentz":
            return rand_lorentz(
                shape,
                std_eta=self.std_eta,
                n_max_std_eta=self.n_max_std_eta,
                device=device,
                dtype=dtype,
            )
        elif self.transform_type == "rotation":
            return rand_rotation(shape, device=device, dtype=dtype)
        elif self.transform_type == "xyrotation":
            return rand_xyrotation(shape, device=device, dtype=dtype)
        elif self.transform_type == "ztransform":
            return rand_ztransform(
                shape,
                std_eta=self.std_eta,
                n_max_std_eta=self.n_max_std_eta,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(
                f"Transformation type {self.transform_type} not implemented"
            )

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        if not self.training:
            frames = Frames(
                is_identity=True,
                shape=fourmomenta.shape[:-1],
                device=fourmomenta.device,
                dtype=fourmomenta.dtype,
            )
            return (frames, {}) if return_tracker else frames

        shape = (
            fourmomenta.shape[:-2] + (1,) if self.is_global else fourmomenta.shape[:-1]
        )
        matrix = self.transform(
            shape, device=fourmomenta.device, dtype=fourmomenta.dtype
        )
        matrix = matrix.expand(*fourmomenta.shape[:-1], 4, 4)

        frames = Frames(
            is_global=self.is_global,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (frames, {}) if return_tracker else frames

    def __repr__(self):
        string = f"RandomFrames(transform_type={self.transform_type}, is_global={self.is_global}"
        if self.transform_type in ["lorentz", "ztransform, general_lorentz"]:
            string += f", std_eta={self.std_eta}"
            string += f", n_max_std_eta={self.n_max_std_eta}"
        string += ")"
        return string


class COMRandomFrames(RandomFrames):
    """Special random frame for data augmentation in amplitude regression.

    Modifies the forward function of RandomFrames such that
    an additional boost is applied to the whole event.
    Only applicable to amplitude regression, the boost changes
    the reference frame to the center of mass of the incoming particles.
    """

    def forward(self, fourmomenta, scalars=None, ptr=None, return_tracker=False):
        if not self.training:
            frames = Frames(
                is_identity=True,
                shape=fourmomenta.shape[:-1],
                device=fourmomenta.device,
                dtype=fourmomenta.dtype,
            )
            return (frames, {}) if return_tracker else frames

        shape = (
            fourmomenta.shape[:-2] + (1,) if self.is_global else fourmomenta.shape[:-1]
        )
        matrix = self.transform(
            shape, device=fourmomenta.device, dtype=fourmomenta.dtype
        )

        # hardcoded for amplitudes
        reference_vector = fourmomenta[..., :2, :].sum(dim=-2, keepdims=True)
        reference_boost = restframe_boost(reference_vector)[..., :4, :4]

        matrix = torch.einsum("...ij,...jk->...ik", matrix, reference_boost)
        matrix = matrix.expand(*fourmomenta.shape[:-1], 4, 4)

        frames = Frames(
            is_global=self.is_global,
            matrices=matrix,
            device=fourmomenta.device,
            dtype=fourmomenta.dtype,
        )
        return (frames, {}) if return_tracker else frames

    def __repr__(self):
        string = f"COMFrames(transform_type={self.transform_type}, is_global={self.is_global}"
        if self.transform_type in ["lorentz", "ztransform", "general_lorentz"]:
            string += f", std_eta={self.std_eta}"
            string += f", n_max_std_eta={self.n_max_std_eta}"
        string += ")"
        return string
