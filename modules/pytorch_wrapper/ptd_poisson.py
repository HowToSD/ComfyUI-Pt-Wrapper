from typing import Any, Dict
import ast
import torch
from torch.distributions import Poisson
import scipy.stats as scst

class PoissonEx(Poisson):
    """
    A class to extend Poisson to add missing functionality.

    pragma: skip_doc
    """
    def cdf(self, x: torch.Tensor):
        """
        Computes the cumulative distribution function (CDF) of the Poisson distribution.

        Args:
            x (torch.Tensor): Input tensor containing values where the CDF is evaluated.

        Returns:
            torch.Tensor: Tensor containing the cumulative probabilities.
        """
        a = x.detach().cpu().numpy()
        rate = self.rate.detach().cpu().numpy()
        outputs = scst.poisson.cdf(a, rate)
        return torch.tensor(outputs, dtype=torch.float32)


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

        dist = PoissonEx(
            rate=torch.tensor(r, dtype=torch.float32))
        return (dist,)
