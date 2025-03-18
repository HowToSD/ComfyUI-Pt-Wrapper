from typing import Any, Dict
import ast
import torch
from torch.distributions import Exponential

class PtdExponential:
    """
    Instantiates a Exponential distribution object.

        Args:
            rate (str): Rate. You can specify a scalar or a tuple of float.
    
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
        Instantiates a Exponential distribution object.
        
        Args:
            rate (str): Probability. You can specify a scalar or a tuple of float.

        Returns:
            tuple: (torch.distributions.binomial.Exponential)
        """
        r = ast.literal_eval(rate)

        dist = Exponential(
            rate=torch.tensor(r))
        return (dist,)
