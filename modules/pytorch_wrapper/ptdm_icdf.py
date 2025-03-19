from typing import Any, Dict
import torch
from torch.distributions import *
import ast

class PtdmIcdf:
    """
    Computes the inverse of the cumulative distribution function for the input distribution.

    **Note**  
    Icdf is not supported for all distributions in PyTorch.

    Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.
            q (float): Value

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
                "distribution": ("PTDISTRIBUTION", {}),
                "q": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, distribution: torch.distributions.Distribution, q: str) -> tuple:
        """
        Computes the inverse of the cumulative distribution function for the input distribution.

        Args:
            distribution (torch.distributions.Distribution): Distribution.
            q (str): Value

        Returns:
            tuple containing the probability
        """
        v2 = ast.literal_eval(q)
        if isinstance(v2, int) is False and isinstance(v2, float) is False:
            if isinstance(distribution, Dirichlet) is False:
                raise ValueError("You need to specify a float or an int.")

        tens = torch.tensor(v2, dtype=torch.float32)

        return (distribution.icdf(tens),)
