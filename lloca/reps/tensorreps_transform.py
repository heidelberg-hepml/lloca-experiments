"""Transforming tensors of different Lorentz group representations."""

import torch

from ..framesnet.frames import Frames
from .tensorreps import TensorReps


class TensorRepsTransform(torch.nn.Module):
    def __init__(
        self,
        reps: TensorReps,
        use_naive=False,
    ):
        """Tensor representation transformation module.

        Parameters
        ----------
        reps: TensorReps
            Tensor representations to transform.
        use_naive: bool
            Whether to use the naive transformation method.
            If False, uses an efficient transformation method.
            Default is False.
        """
        super().__init__()
        self.reps = reps
        self.transform = (
            self._transform_naive if use_naive else self._transform_efficient
        )

        # cache idx_start and idx_end for each rep
        self.start_end_idx = []
        idx = 0
        for mul_rep in self.reps:
            _, rep = mul_rep
            self.start_end_idx.append([idx, idx + mul_rep.dim])
            idx += mul_rep.dim

        # build parity_mask
        parity_odd = torch.zeros(self.reps.dim, dtype=torch.bool)
        idx = 0
        for mul_rep in self.reps:
            _, rep = mul_rep
            parity_odd[idx : idx + mul_rep.dim] = True if rep.parity == -1 else False
            idx += mul_rep.dim
        self.register_buffer("parity_odd", parity_odd.unsqueeze(0))
        self.no_parity_odd = self.parity_odd.sum() == 0

        if not use_naive:
            # build mapping from order to element in the reps list
            # only used in _transform_efficient
            self.map_rep = [None for _ in range(self.reps.max_rep.rep.order + 1)]
            idx_rep = 0
            for i in range(self.reps.max_rep.rep.order + 1):
                if self.reps[idx_rep].rep.order == i:
                    self.map_rep[i] = idx_rep
                    idx_rep += 1

        if self.reps.max_rep.rep.order <= 1:
            # super efficient shortcut if only scalar and vector reps are present
            self.transform = self._transform_only_scalars_and_vectors

    def forward(self, tensor: torch.Tensor, frames: Frames):
        """Apply a transformation to a tensor of a given representation.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to transform, shape (..., self.reps.dim).
        frames: Frames
            The local frames to apply the transformation with, shape (..., 4, 4).

        Returns
        -------
        torch.Tensor
            The transformed tensor, shape (..., self.reps.dim).
        """
        assert (
            self.reps.dim == tensor.shape[-1]
        ), f"Last tensor dimension is {tensor.shape[-1]}, but should be reps.dim={self.reps.dim}."

        if frames.is_identity or (
            self.reps.mul_without_scalars == 0 and self.no_parity_odd
        ):
            return tensor

        in_shape = tensor.shape
        if len(frames.shape) > 3:
            frames = frames.reshape(-1, 4, 4)
        tensor = tensor.reshape(-1, tensor.shape[-1])
        assert (
            tensor.shape[0] == frames.shape[0]
        ), f"Batch dimension is {tensor.shape[0]} for tensor, but {frames.shape[0]} for frames."

        tensor_transformed = self.transform(tensor, frames)
        tensor_transformed = self.transform_parity(tensor_transformed, frames)

        tensor_transformed = tensor_transformed.reshape(*in_shape)
        return tensor_transformed

    def _transform_naive(self, tensor, frames):
        """Naive transform: Apply n transformations to a tensor of n'th order.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to transform, shape (N, self.reps.dim).
        frames: Frames
            The local frames to apply the transformation with, shape (N, 4, 4).

        Returns
        -------
        torch.Tensor
            The transformed tensor, shape (N, self.reps.dim).
        """
        output = tensor.clone()
        frames = frames.matrices.clone().to(tensor.dtype)
        for mul_rep, [idx_start, idx_end] in zip(self.reps, self.start_end_idx):
            mul, rep = mul_rep
            if mul == 0 or rep.order == 0:
                continue

            x = tensor[:, idx_start:idx_end].reshape(-1, mul, *([4] * rep.order))

            einsum_string = get_einsum_string(rep.order)
            print(frames.shape, frames.matrices.to(x.dtype).shape)
            x_transformed = torch.einsum(einsum_string, *([frames] * rep.order), x)
            output[:, idx_start:idx_end] = x_transformed.reshape(-1, mul_rep.dim)

        return output

    def _transform_efficient(self, tensor, frames):
        """Efficient transform:
        Starting with the highest-order tensor contribution,
        add the next contribution, apply frames transformation
        and flatten first dimension before continueing with next order.

        This is more efficient, because we use the
        maximum amount of parallelization possible.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to transform, shape (N, self.reps.dim).
        frames: Frames
            The local frames to apply the transformation with, shape (N, 4, 4).

        Returns
        -------
        torch.Tensor
            The transformed tensor, shape (N, self.reps.dim).
        """
        output = None
        bframes = frames.matrices.clone().to(tensor.dtype)
        for order in reversed(range(self.reps.max_rep.rep.order + 1)):
            if self.map_rep[order] is not None:
                # add new contribution to the mix
                idx_start, idx_end = self.start_end_idx[self.map_rep[order]]
                contribution = tensor[:, idx_start:idx_end].reshape(
                    tensor.shape[0], -1, *(order * (4,))
                )
                output = (
                    torch.cat([contribution, output], dim=1)
                    if order < self.reps.max_rep.rep.order
                    else contribution
                )

            if order > 0:
                # apply transformation, then flatten because transformation is done
                output = torch.einsum("ijk,ilk...->ilj...", bframes, output)
                output = output.flatten(start_dim=1, end_dim=2)

        return output

    def _transform_only_scalars_and_vectors(self, tensor, frames):
        """Super efficient transform that assumes that only scalars and vectors are present.
        Follows the same recipe as _transform_efficient, but avoids small overheads from
        torch.cat and torch.reshape.

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to transform, shape (N, self.reps.dim).
        frames: Frames
            The local frames to apply the transformation with, shape (N, 4, 4).

        Returns
        -------
        torch.Tensor
            The transformed tensor, shape (N, self.reps.dim).
        """
        if self.reps.max_rep.rep.order == 0:
            return tensor

        output = tensor.clone()
        vector_idx_start, vector_idx_end = self.start_end_idx[-1]
        vectors = tensor[:, vector_idx_start:vector_idx_end]
        vectors = vectors.reshape(tensor.shape[0], -1, 4)
        vectors_transformed = torch.einsum(
            "ijk,ilk->ilj",
            frames.matrices.to(vectors.dtype),
            vectors,
        )
        output[:, vector_idx_start:vector_idx_end] = vectors_transformed.reshape(
            tensor.shape[0], -1
        )
        return output

    def transform_parity(self, tensor, frames):
        """Parity transform: Multiply parity-odd states by sign(det Lambda).

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to transform, shape (N, self.reps.dim).
        frames: Frames
            The local frames to apply the transformation with, shape (N, 4, 4).

        Returns
        -------
        torch.Tensor
            The transformed tensor, shape (N, self.reps.dim).
        """
        if self.no_parity_odd:
            return tensor
        else:
            return torch.where(
                self.parity_odd, frames.det.sign().unsqueeze(-1) * tensor, tensor
            )


def get_einsum_string(order):
    """Create einsum string for transformation of order-n tensor in _transform_naive.

    Parameters
    ----------
    order: int
        The order of the tensor representation.

    Returns
    -------
    str
        The einsum string for the transformation.
    """
    if order > 12:
        raise NotImplementedError("Running out of letters for order>12")

    einsum = ""
    start = ord("A")
    batch_index = ord("a")

    # list of frames
    for i in range(order):
        einsum += chr(batch_index) + chr(start + 2 * i) + chr(start + 2 * i + 1) + ","

    # tensor
    einsum += chr(batch_index)
    einsum += chr(start + 2 * order + 1)
    for i in range(order):
        einsum += chr(start + 2 * i + 1)

    # output
    einsum += "->"
    einsum += chr(batch_index)
    einsum += chr(start + 2 * order + 1)
    for i in range(order):
        einsum += chr(start + 2 * i)

    return einsum
