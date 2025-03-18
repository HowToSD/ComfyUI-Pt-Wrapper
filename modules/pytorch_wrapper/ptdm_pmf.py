from typing import Any, Dict
import torch
from torch.distributions import *
import ast

class PtdmPmf:
    """
    Computes the probability for the input distribution.

    Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.
            k (int): Number of successes.

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
                "k": ("INT", {"default": 0, "min": -2**31, "max": 2**31})
            }
        }

    RETURN_TYPES: tuple = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, distribution: torch.distributions.Distribution, k: int) -> tuple:
        """
        Computes the probability for the input distribution.

        Args:
            distribution (torch.distributions.Distribution): Distribution.
            k (int): Number of successes.

        Returns:
            tuple containing the probability
        """
        if isinstance(distribution, (Poisson, Categorical, Multinomial, Binomial, NegativeBinomial)):
            tens = torch.tensor(k, dtype=torch.int64)
        else:
            tens = torch.tensor(k, dtype=torch.float32)

        return (torch.exp(distribution.log_prob(tens)),)
