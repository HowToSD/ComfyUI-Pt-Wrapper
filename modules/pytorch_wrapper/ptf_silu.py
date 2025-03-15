from typing import Any, Dict, Callable, Tuple
import torch.nn.functional


class PtfSiLU:
    """
    Ptf SiLU:

    A PyTorch wrapper for the SiLU activation function.

    This class provides a callable that applies the SiLU activation function 
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
        Returns the PyTorch SiLU function as a callable.

        Returns:
            Tuple[Callable]: A tuple containing `torch.nn.functional.silu`.
        """
        return (torch.nn.functional.silu,)
