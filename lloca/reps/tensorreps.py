"""Lorentz group representations."""

from typing import Tuple


class TensorRep(Tuple):
    """Individual tensor representation"""

    def __new__(cls, order, parity):
        """Create a tensor representation based on its order and parity.

        Parameters
        ----------
        order: int
            The order of the tensor representation.
        parity: int
            The parity of the tensor representation. Can be -1 or 1
        """
        assert (
            isinstance(order, int) and order >= 0
        ), f"order must be a non-negative integer, but got {order}"
        assert parity in [
            -1,
            1,
        ], f"parity must be either -1 (p) or 1 (n), but got {parity}"
        return super().__new__(cls, (order, parity))

    def __deepcopy__(self, memo):
        return self

    @property
    def order(self):
        """The order of the tensor."""
        return self[0]

    @property
    def parity(self):
        """The parity of the tensor."""
        return self[1]

    def __repr__(self):
        """Returns a string representation of the tensor."""
        return f"{self.order}{'n' if self.parity == 1 else 'p'}"


class _TensorMulRep(Tuple):
    """Direct product of similar tensor representations"""

    def __new__(cls, mul, rep):
        """Create a tensor multiplication representation.
        This represents a direct product of a tensor representation with a multiplier.

        Parameters
        ----------
        mul: int
            The multiplier for the tensor representation.
        rep: TensorRep
            The tensor representation.
        """

        assert (
            isinstance(mul, int) and mul >= 0
        ), f"mul must be a non-negative integer, but got {mul}"
        assert isinstance(
            rep, TensorRep
        ), f"rep must be an instance of TensorRep, but got type {type(rep)}"

        return super().__new__(cls, (mul, rep))

    def __deepcopy__(self, memo):
        return self

    @property
    def mul(self):
        """The multiplier for the tensor representation."""
        return self[0]

    @property
    def rep(self):
        """The tensor representation."""
        return self[1]

    @property
    def dim(self):
        """The dimension of the tensor multiplication."""
        return (4**self.rep.order) * self.mul

    def __repr__(self):
        """Returns a string representation of the tensor multiplication."""
        return f"{self.mul}x{self.rep}"


class TensorReps(Tuple):
    """Generic tensor representations"""

    def __new__(cls, input, simplify=True):
        """Create a tensor representation based on the input.
        This can be a string representation or a list of _TensorMulRep instances.

        Parameters
        ----------
        input: Union[TensorReps, str, List[_TensorMulRep]]
            The input to initialize the tensor reps.
            If `input` is a string, it is parsed to extract the tensor representations.
            If `input` is a list, it should contain instances of `_TensorMulRep`.
        simplify: bool
            Whether to simplify the tensor reps after initialization. Default is True.
        """
        if isinstance(input, TensorReps):
            tensor_reps = input
        elif isinstance(input, list | tuple):
            assert all(isinstance(x, _TensorMulRep) for x in input)
            tensor_reps = input
        elif isinstance(input, str):
            try:
                tensor_reps = parse_tensorreps_string(input)
            except Exception:
                raise ValueError(f"Invalid tensor_reps string {input}")
        else:
            raise ValueError(f"Invalid input: {input} is of type {type(input)}")

        ret = super().__new__(cls, tensor_reps)
        if simplify:
            return ret.simplify()
        return ret

    def __repr__(self):
        """Returns a string representation of the tensor reps."""
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def __deepcopy__(self, memo):
        return self

    @property
    def dim(self):
        """The total dimension of the tensor reps."""
        return sum(mul_ir.dim for mul_ir in self)

    @property
    def max_rep(self):
        """The tensor irrep with the highest order."""
        return max(self, key=lambda x: x.rep.order)

    @property
    def mul_without_scalars(self):
        """The total multiplier of the tensor reps without the scalars."""
        return sum(mul_ir.mul for mul_ir in self if mul_ir.rep.order != 0)

    @property
    def mul_scalars(self):
        """The total multiplier of the tensor reps scalars."""
        return sum(mul_ir.mul for mul_ir in self if mul_ir.rep.order == 0)

    @property
    def mul(self):
        """The total multiplier of the tensor reps."""
        return sum(mul_ir.mul for mul_ir in self)

    @property
    def reps(self):
        """The set of tensor reps."""
        return {rep for _, rep in self}

    @property
    def is_sorted(self):
        """Whether the tensor reps are sorted by the order of the reps."""
        if len(self) <= 1:
            return True
        else:
            return all(
                self[i].rep.order <= self[i + 1].rep.order for i in range(len(self) - 1)
            )

    def __add__(self, tensor_reps, simplify=True):
        """Adds tensor reps to the current tensor reps."""
        tensor_reps = TensorReps(tensor_reps)
        return TensorReps(super().__add__(tensor_reps), simplify=simplify)

    def sort(self):
        """Sorts the tensor reps by the order of the reps."""
        return TensorReps(sorted(self, key=lambda x: x.rep.order))

    def simplify(self):
        """Simplifies the tensor reps by combining the same reps."""
        items = self if self.is_sorted else self.sort()

        out = []
        for mul_rep in items:
            mul, rep = mul_rep
            if len(out) > 0 and out[-1].rep == rep:
                # same rep -> extend mul of previous rep
                out[-1] = _TensorMulRep(out[-1].mul + mul, rep)
            elif mul > 0:
                # different rep and mul>0 -> create new entry
                out.append(mul_rep)

        return TensorReps(out, simplify=False)

    def is_simplified(self):
        """Check if the TensorReps is simlified."""
        if not self.is_sorted:
            return False

        return all(
            self[i].rep.order != self[i + 1].rep.order for i in range(len(self[:-1]))
        )


def parse_tensorreps_string(input):
    """Parse a string representation of tensor representations into a list of _TensorMulRep instances.

    Parameters
    ----------
    input: str
        The string representation of tensor representations, e.g. "2x2n+3x1p+1x0n".

    Returns
    -------
    List[_TensorMulRep]
        A list of _TensorMulRep instances representing the tensor representations.
    """
    out = []
    input = input.replace(" ", "")  # remove whitespace
    input_list = input.split("+")  # split into single _TensorMulRep
    for i, rep in enumerate(input_list):
        if rep[-1] == "n":
            parity = 1
            rep = rep[:-1]
        elif rep[-1] == "p":
            parity = -1
            rep = rep[:-1]
        else:
            raise ValueError(
                f"Invalid last character (=parity) in tensorreps string {rep}, should be either 'n' or 'p'"
            )

        mul, order = rep.split("x")
        mul, order = int(mul), int(order)
        out.append(_TensorMulRep(mul, TensorRep(order, parity)))
    return out
