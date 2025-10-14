"""Generic LLoCa MessagePassing module."""
from typing import Any, Dict
import torch
from torch_geometric.nn import MessagePassing

from ..framesnet.frames import ChangeOfFrames, Frames, IndexSelectFrames
from ..reps.tensorreps_transform import TensorRepsTransform


class LLoCaMessagePassing(MessagePassing):
    """Adaptation of the torch_geometric MessagePassing class using the LLoCa formalism."""

    def __init__(self, params_dict: Dict[str, Dict[str, Any]], aggr="add") -> None:
        """Initializes a new instance of the LLoCaMessagePassing class.

        Parameters
        ----------
        params_dict: Dict[str, Dict[str, Any]]
            A dictionary containing the parameters for the message passing algorithm and the corresponding representations.
            Each key in the dictionary represents a feature, and the value is another dictionary with keys "type" and "rep".
            The "type" can be either "local" or "global", and "rep" is an instance of TensorRepsTransform that defines how to transform the features.
        aggr: str, optional
            The aggregation method to use for combining messages. Defaults to "add".
        """
        super().__init__(aggr=aggr)

        self.params_dict = params_dict

        tmp_dict = {}

        for key, value in self.params_dict.items():
            if value["type"] is not None:
                tmp_dict[key] = TensorRepsTransform(value["rep"])

        self.transform_dict = torch.nn.ModuleDict(tmp_dict)

        # Register hooks to call before propagating and before sending messages
        self.register_propagate_forward_pre_hook(self.pre_propagate_hook)
        self.register_message_forward_pre_hook(self.pre_message_hook)

    def pre_propagate_hook(self, module: Any, inputs: tuple) -> tuple:
        """A hook method called before propagating messages in the message passing algorithm. We
        save the frames in the class variable and remove it from the inputs dictionary.
        """
        assert (
            inputs[-1].get("frames") is not None
        ), "frames are not in the propagate inputs"

        self._frames = inputs[-1]["frames"]
        self._edge_index = inputs[0]

        return inputs

    def pre_message_hook(self, module: Any, inputs: tuple) -> tuple:
        """Pre-message hook method that is called before passing messages in the message passing
        algorithm. We transform the features according to the representations in the params_dict.
        """

        # calculate frames_i, frames_j and the U matrix
        if isinstance(self._frames, tuple):
            frames_i = IndexSelectFrames(self._frames[1], self._edge_index[1])
            frames_j = IndexSelectFrames(self._frames[0], self._edge_index[0])
        elif isinstance(self._frames, Frames):
            frames_i = IndexSelectFrames(self._frames, self._edge_index[1])
            frames_j = IndexSelectFrames(self._frames, self._edge_index[0])
        else:
            raise ValueError(
                f"frames should be either a tuple or an Frames object but is {type(self._frames)}"
            )

        U = ChangeOfFrames(frames_start=frames_j, frames_end=frames_i)

        # now go through the params_dict and get the representations and transform the features in the right way
        for param, param_info in self.params_dict.items():
            if param_info["type"] == "local":
                assert param + "_j" in inputs[-1], f"Key {param}_j not in inputs"
                # transform the features according to the representation
                inputs[-1][param + "_j"] = self.transform_dict[param](
                    inputs[-1][param + "_j"], U
                )
            elif param_info["type"] == "global":
                if inputs[-1].get(param) is not None:
                    inputs[-1][param] = self.transform_dict[param](
                        inputs[-1][param], frames_i
                    )
                if inputs[-1].get(param + "_j") is not None:
                    inputs[-1][param + "_j"] = self.transform_dict[param](
                        inputs[-1][param + "_j"], frames_i
                    )
                if inputs[-1].get(param + "_i") is not None:
                    inputs[-1][param + "_i"] = self.transform_dict[param](
                        inputs[-1][param + "_i"], frames_i
                    )

        return inputs
