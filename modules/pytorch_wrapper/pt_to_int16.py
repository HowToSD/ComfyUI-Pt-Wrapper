from typing import Any, Dict
import torch

class PtToInt16:
    """
    Pt To Int16:
    Converts the input tensor's data type to int16.

    category: PyTorch wrapper - Data type conversion
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
                "tens_a": ("TENSOR", {})
            }
        }

    RETURN_TYPES: tuple = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Data Analysis"

    def f(self, tens_a: torch.Tensor) -> tuple:
        """
        Converts the input tensor's data type to int16.

        Args:
            tens_a (torch.Tensor): A PyTorch tensor of numerical type.

        Returns:
            tuple: A tuple containing the resultant tensor after the operation.
        """
        return (tens_a.to(torch.int16),)
