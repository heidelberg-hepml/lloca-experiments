import inspect
from contextlib import contextmanager

import torch


@contextmanager
def track_clamps():
    clamp_calls = []

    original_clamp_fn = torch.clamp
    original_tensor_clamp = torch.Tensor.clamp
    original_tensor_clamp_ = torch.Tensor.clamp_

    def record_clamp_call(op_type, before, after):
        stack = inspect.stack()
        for frame in stack[2:]:
            if "torch" not in frame.filename:
                diff = (before != after).sum().item()
                return {
                    "filename": frame.filename,
                    "line": frame.lineno,
                    "function": frame.function,
                    "code": frame.code_context[0].strip() if frame.code_context else None,
                    "op_type": op_type,
                    "num_elements_clamped": diff,
                    "total_elements": before.numel(),
                }

    def tracking_clamp(*args, **kwargs):
        input_tensor = args[0]
        before = input_tensor.clone()
        result = original_clamp_fn(*args, **kwargs)
        callsite = record_clamp_call("torch.clamp", before, result)
        if callsite:
            clamp_calls.append(callsite)
        return result

    def tracking_tensor_clamp(self, *args, **kwargs):
        before = self.clone()
        result = original_tensor_clamp(self, *args, **kwargs)
        callsite = record_clamp_call("tensor.clamp", before, result)
        if callsite:
            clamp_calls.append(callsite)
        return result

    def tracking_tensor_clamp_(self, *args, **kwargs):
        before = self.clone()
        original_tensor_clamp_(self, *args, **kwargs)
        after = self
        callsite = record_clamp_call("tensor.clamp_", before, after)
        if callsite:
            clamp_calls.append(callsite)
        return self

    torch.clamp = tracking_clamp
    torch.Tensor.clamp = tracking_tensor_clamp
    torch.Tensor.clamp_ = tracking_tensor_clamp_

    try:
        yield clamp_calls  # gives results
    finally:
        # Restore originals
        torch.clamp = original_clamp_fn
        torch.Tensor.clamp = original_tensor_clamp
        torch.Tensor.clamp_ = original_tensor_clamp_
