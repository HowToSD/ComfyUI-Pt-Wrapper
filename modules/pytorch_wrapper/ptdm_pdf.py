from typing import Any, Dict
import torch
from torch.distributions import *
import ast

class PtdmPdf:
    """
    Computes the probability density for the input distribution.

    Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.
            x (float): Value

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
                "x": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, distribution: torch.distributions.Distribution, x: str) -> tuple:
        """
        Computes the probability for the input distribution.

        Args:
            distribution (torch.distributions.Distribution): Distribution.
            x (str): Value

        Returns:
            tuple containing the probability
        """
        v2 = ast.literal_eval(x)
        if isinstance(v2, int) is False and isinstance(v2, float) is False :
            if isinstance(distribution, Dirichlet) is False:
                raise ValueError("You need to specify a float or an int.")

        if isinstance(distribution, (Poisson, Categorical, Multinomial, Binomial, NegativeBinomial)):
            tens = torch.tensor(v2, dtype=torch.int64)
        else:
            tens = torch.tensor(v2, dtype=torch.float32)

        return (torch.exp(distribution.log_prob(tens)),)
