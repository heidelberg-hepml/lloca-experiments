"""Bookkeeping classes to access local frames."""
import torch

from ..utils.lorentz import lorentz_eye, lorentz_metric


class Frames:
    """
    Bookkeeping class for local frames.

    Collection of Lorentz transformations, represented as (..., 4, 4) matrices.
    Properties like det and inv are cached for performance.
    Attributes should not be changed after initialization to avoid inconsistencies.
    The object can be modified with e.g. ``.reshape()``, ``.expand()`` and ``.repeat()``, ``.to()``.
    """

    def __init__(
        self,
        matrices: torch.Tensor = None,
        is_global: bool = False,
        det: torch.Tensor = None,
        inv: torch.Tensor = None,
        is_identity: bool = False,
        shape=None,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        """There are 2 ways to initialize an Frames object, with different arguments:
        - From matrices: Set matrices and optionally is_global, det, inv
        - As identity: Set is_identity=True, shape, device, dtype

        Parameters
        ----------
        matrices: torch.tensor
            Transformation matrices of shape (..., 4, 4)
        is_global: bool
            Whether frames are the same for all particles in the point cloud
        inv: torch.Tensor
            Optional cached inverse of shape (..., 4, 4).
            If not given, takes a bit of extra time to compute.
        det: torch.Tensor
            Optional cached determinant of shape (...).
            If not given, takes a bit of extra time to compute.
        is_identity: bool
            Sets matrices to diagonal.
        shape: List[int]
            Specifies matrices.shape[:-2] if is_identity. Otherwise inferred from matrices.
        device: str
            Specifies device if is_identity. Otherwise inferred from matrices.
        dtype: torch.dtype
            Specifies dtype if is_identity. Otherwise inferred from matrices.
        """
        # straight-forward initialization
        self.is_identity = is_identity
        if is_identity:
            if matrices is None:
                assert shape and device and dtype
            else:
                shape = matrices.shape[:-2]
                device = matrices.device
                dtype = matrices.dtype

            self.matrices = lorentz_eye(shape, device=device, dtype=dtype)
            self.is_global = True
            self.det = torch.ones(self.shape[:-2], dtype=self.dtype, device=self.device)
            self.inv = self.matrices
        else:
            assert matrices is not None
            assert matrices.shape[-2:] == (
                4,
                4,
            ), f"matrices must be of shape (..., 4, 4), but found {matrices.shape[-2:]} instead"

            self.matrices = matrices
            self.is_global = is_global
            if det is not None:
                assert det.shape == matrices.shape[:-2]
            if inv is not None:
                assert inv.shape == matrices.shape
            self.det = det
            self.inv = inv

        # cache expensive properties
        self.metric = lorentz_metric(
            self.shape[:-2], device=self.device, dtype=self.dtype
        )
        if self.det is None:
            self.det = torch.linalg.det(self.matrices)
        if self.inv is None:
            self.inv = self.metric @ self.matrices.transpose(-1, -2) @ self.metric

    def reshape(self, *shape):
        """Reshape the matrices to generate a new object of different shape.

        Parameters
        ----------
        shape: int
            New shape for the matrices. The last two dimensions must be (4, 4).

        Returns
        -------
        Frames
        """
        assert shape[-2:] == (4, 4)
        return Frames(
            matrices=self.matrices.reshape(*shape),
            is_identity=self.is_identity,
            is_global=self.is_global,
            inv=self.inv.reshape(*shape),
            det=self.det.reshape(*shape[:-2]),
        )

    def expand(self, *shape):
        """Expand the matrices to generate a new object of different shape.

        Parameters
        ----------
        shape: int
            New shape for the matrices. The last two dimensions must be (4, 4).

        Returns
        -------
        Frames
        """
        assert shape[-2:] == (4, 4)
        return Frames(
            matrices=self.matrices.expand(*shape),
            is_identity=self.is_identity,
            is_global=self.is_global,
            inv=self.inv.expand(*shape),
            det=self.det.expand(*shape[:-2]),
        )

    def repeat(self, *shape):
        """Repeat the matrices to generate a new object of different shape.

        Parameters
        ----------
        shape: int
            New shape for the matrices. The last two dimensions must be (4, 4).

        Returns
        -------
        Frames
        """
        assert shape[-2:] == (1, 1)
        return Frames(
            matrices=self.matrices.repeat(*shape),
            is_identity=self.is_identity,
            is_global=self.is_global,
            inv=self.inv.repeat(*shape),
            det=self.det.repeat(*shape[:-2]),
        )

    def to(self, dtype=None, device=None):
        """Move the matrices to a new device and/or dtype."""
        self.matrices = self.matrices.to(device=device, dtype=dtype)
        self.inv = self.inv.to(device=device, dtype=dtype)
        self.det = self.det.to(device=device, dtype=dtype)

    def __repr__(self):
        return repr(self.matrices)

    @property
    def device(self):
        return self.matrices.device

    @property
    def dtype(self):
        return self.matrices.dtype

    @property
    def shape(self):
        return self.matrices.shape


class InverseFrames(Frames):
    """Inverse of a collection of frames."""

    def __init__(self, frames: Frames):
        super().__init__(
            matrices=frames.inv,
            is_global=frames.is_global,
            inv=frames.matrices,
            det=frames.det,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.shape,
        )


class IndexSelectFrames(Frames):
    """Index-controlled collection of frames."""

    def __init__(self, frames: Frames, indices: torch.Tensor):
        super().__init__(
            matrices=frames.matrices.index_select(0, indices),
            is_global=frames.is_global,
            inv=frames.inv.index_select(0, indices),
            det=frames.det.index_select(0, indices),
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.shape,
        )


class ChangeOfFrames(Frames):
    """
    Change of frames between two Frames objects.

    Formally, for L_start and L_end we have $L_change = L_end * L_start^{-1}$.

    WARNING: This function does not mix the frames of different point clouds.
    It is used in TFMessagePassing where this mixing is performed using the edge_index.
    """

    def __init__(self, frames_start: Frames, frames_end: Frames):
        assert frames_start.shape == frames_end.shape
        if frames_start.is_global:
            super().__init__(
                is_identity=True,
                shape=frames_start.shape,
                device=frames_start.device,
                dtype=frames_start.dtype,
            )
        else:
            super().__init__(
                matrices=frames_end.matrices @ frames_start.inv,
                is_global=False,
                inv=frames_start.matrices @ frames_end.inv,
                det=frames_start.det * frames_end.det,
            )


class LowerIndicesFrames(Frames):
    """
    Frames with lower indices, obtained by multiplying with the metric.
    Used in LLoCaAttention to lower the key indices.
    """

    def __init__(self, frames):
        super().__init__(
            matrices=frames.metric @ frames.matrices,
            inv=frames.inv @ frames.metric,
            det=-frames.det,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames.shape,
        )
