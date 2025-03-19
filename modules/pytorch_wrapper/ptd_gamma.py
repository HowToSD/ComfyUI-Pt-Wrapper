from typing import Any, Dict
import ast
import torch
from torch.distributions import Gamma
import scipy.stats as scst


class GammaEx(Gamma):
    """
    A class to extend Gamma to add missing functionality.

    pragma: skip_doc
    """

    def icdf(self, q: torch.Tensor):
        """
        Computes the inverse of cumulative distribution function (CDF) of the Gamma distribution.

        Args:
            q (torch.Tensor): Input tensor containing values where the ICDF is evaluated.

        Returns:
            torch.Tensor: Tensor containing the inverse of cumulative probabilities.
        """
        x2 = q.detach().cpu().numpy()
        a = self.concentration.detach().cpu().item()
        b = self.rate.detach().cpu().item()
        outputs = scst.gamma.ppf(x2, a, 1/b)  # Note that you need to pass the inverse of b.
        return torch.tensor(outputs, dtype=torch.float32).to(q.device)


class PtdGamma:
    """
    Instantiates a Gamma distribution object.

        Args:
            alpha: Shape. You can specify a scalar or a tuple of float.  
            beta: Rate. You can specify a scalar or a tuple of float.  
    
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
                "alpha": ("STRING", {"default": ""}),
                "beta": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("PTDISTRIBUTION",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, alpha: str, beta: str) -> tuple:
        """
        Instantiates a Gamma distribution object.
        
        Args:
            alpha (str): Concentration You can specify a scalar or a tuple of float.  
            beta (str): Rate. You can specify a scalar or a tuple of float.  

        Returns:
            tuple: (torch.distributions.binomial.Gamma)
        """
        c = ast.literal_eval(alpha)
        r = ast.literal_eval(beta)

        dist = GammaEx(
            concentration=torch.tensor(c, dtype=torch.float32),
            rate=torch.tensor(r, dtype=torch.float32))
        return (dist,)
