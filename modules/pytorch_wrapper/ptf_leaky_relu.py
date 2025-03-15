from typing import Any, Dict, Callable, Tuple
import torch.nn.functional

import functools

class PtfLeakyReLU:
    """
    Ptf LeakyReLU:

    A PyTorch wrapper for the LeakyReLU activation function.

    This class provides a callable that applies the LeakyReLU activation function 
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
        return {
            "required": {
                "negative_slope": ("FLOAT", {"default": 0.01, "min": 1e-6, "max": 1e4, "step": 1e-5}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PTCALLABLE",)
    FUNCTION: str = "f"
    CATEGORY: str = "Data Analysis"

    def f(self, negative_slope: float) -> Tuple[Callable]:
        """
        Returns the PyTorch LeakyReLU function as a callable.

        Args:
            negative_slope (float): The multiplier for the negative input.
        Returns:
            Tuple[Callable]: A tuple containing `torch.nn.functional.leaky_relu`.
        """
        leaky_relu_last_axis = functools.partial(torch.nn.functional.leaky_relu, negative_slope=negative_slope)

        return (leaky_relu_last_axis,)
