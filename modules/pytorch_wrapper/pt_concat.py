from typing import Any, Dict
import torch

class PtConcat:
    """
    Concatenates two PyTorch tensors.

    category: PyTorch wrapper - Transform
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the required input types.

        Returns:
            Dict[str, Any]: A dictionary of required input types.
        """
        return {
            "required": {
                "tens_a": ("TENSOR", {}),
                "tens_b": ("TENSOR", {}),
                "dim": ("INT", {"default":0, "min":-10, "max":10})
            }
        }

    RETURN_TYPES: tuple = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Data Analysis"

    def f(self, tens_a: torch.Tensor, tens_b: torch.Tensor, dim: int) -> tuple:
        """
        Concatenates two PyTorch tensors.

        Args:
            tens_a (torch.Tensor): A PyTorch tensor of any shape.
            tens_b (torch.Tensor): A PyTorch tensor of shape that supports concatenation with `tens_a`.
            dim (int): An axis to concatenate on.
        Returns:
            tuple: A tuple containing the resultant tensor.
        """
        return (torch.concat([tens_a, tens_b], dim=dim),)
