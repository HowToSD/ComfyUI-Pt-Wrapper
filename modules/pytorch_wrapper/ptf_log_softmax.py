from typing import Any, Dict, Callable, Tuple
import torch
import torch.nn.functional
import functools

log_softmax_last_axis = functools.partial(torch.nn.functional.log_softmax, dim=-1)

class PtfLogSoftmax:
    """
    The log softmax activation function.

    This class provides a callable that applies the log_softmax activation function 
    from `torch.nn.functional`.

    category: PyTorch wrapper - Callable
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the required input types.

        Returns:
            Dict[str, Any]: An empty dictionary since this function requires no inputs.
        """
        return {}

    RETURN_TYPES: Tuple[str, ...] = ("PTCALLABLE",)
    FUNCTION: str = "f"
    CATEGORY: str = "Data Analysis"

    def f(self) -> Tuple[Callable]:
        """
        Returns the PyTorch log_softmax function as a callable.

        Returns:
            Tuple[Callable]: A tuple containing `logsoftmax_last_axis`.
        """
        return (log_softmax_last_axis,)
