from typing import Any, Dict
import ast
import torch
from torch.distributions import Poisson

class PtdPoisson:
    """
    Instantiates a Poisson distribution object.

        Args:
            rate (str): Rate. You can specify a scalar or a tuple of int.
    
    category: PyTorch wrapper - Distribution
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types for the function.

        Returns:
            Dict[str, Any]: A dictionary specifying the required input types.
        """
        return {
            "required": {
                "rate": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("PTDISTRIBUTION",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, rate: str) -> tuple:
        """
        Instantiates a Poisson distribution object.
        
        Args:
            rate (str): Probability. You can specify a scalar or a tuple of int.

        Returns:
            tuple: (torch.distributions.poisson.Poisson)
        """
        r = ast.literal_eval(rate)

        dist = Poisson(
            rate=torch.tensor(r, dtype=torch.float32))
        return (dist,)
