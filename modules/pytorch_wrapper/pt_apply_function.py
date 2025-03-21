from typing import Any, Dict, Tuple, Callable
import torch


class PtApplyFunction:
    """
    Applies a function to the input tensor.
    
    ### Fields:  
    - tens: Tensor to which the function be applied.
    - closure: Differentiable function to apply to the tensor.

    category: PyTorch wrapper - Training
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the expected input types.

        Returns:
            Dict[str, Any]: A dictionary specifying required and optional input types.
        """
        return {
            "required": {
                "tens": ("TENSOR", {}),
                "closure": ("PTCALLABLE", {})
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self,
          tens: torch.Tensor,
          closure: Callable) -> Tuple[torch.Tensor]:
        """
        Applies a function to the input tensor.

        Args:
            tens (torch.Tensor): Tensor to which the function be applied.
            closure (Callable): Differentiable function to apply to the tensor.

        Returns:
            Tuple[nn.Module]: A tuple containing the tensor.
        """
        return (closure(tens),)
