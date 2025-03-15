from typing import Any, Dict, Callable, Tuple
import torch.nn.functional
import functools

softmax_last_axis = functools.partial(torch.nn.functional.softmax, dim=-1)

class PtfSoftmax:
    """
    A PyTorch wrapper for the softmax activation function.

    This class provides a callable that applies the softmax activation function 
    from `torch.nn.functional`.

    Internally, this node calls torch.nn.functional.softmax(dim=-1).
    See https://github.com/pytorch/pytorch/issues/1020 to find out more about the meaning of dim.

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
        Returns the PyTorch softmax function as a callable.

        Returns:
            Tuple[Callable]: A tuple containing `torch.nn.functional.relu`.
        """
        return (softmax_last_axis,)
