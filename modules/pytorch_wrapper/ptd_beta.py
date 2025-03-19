from typing import Any, Dict
import ast
import torch
from torch.distributions import Beta
import scipy.stats as scst

class BetaEx(Beta):
    """
    A class to extend Beta to add missing functionality.

    pragma: skip_doc
    """
    def cdf(self, x: torch.Tensor):
        """
        Computes the cumulative distribution function (CDF) of the Beta distribution.

        Args:
            x (torch.Tensor): Input tensor containing values where the CDF is evaluated.

        Returns:
            torch.Tensor: Tensor containing the cumulative probabilities.
        """
        x2 = x.detach().cpu().numpy()
        a = self.concentration1.detach().cpu().item()
        b = self.concentration0.detach().cpu().item()
        outputs = scst.beta.cdf(x2, a, b)
        return torch.tensor(outputs, dtype=torch.float32).to(x.device)


class PtdBeta:
    """
    Instantiates a Beta distribution object.

        Args:
            alpha: The alpha parameter of the distsribution. You can specify a scalar or a tuple of float.  
            beta: The beta parameter of the distribution. You can specify a scalar or a tuple of float.  
    
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
        Instantiates a Beta distribution object.
        
        Args:
            alpha: The alpha parameter of the distsribution. You can specify a scalar or a tuple of float.  
            beta: The beta parameter of the distribution. You can specify a scalar or a tuple of float.  

        Returns:
            tuple: (torch.distributions.binomial.Beta)
        """
        c1 = ast.literal_eval(alpha)
        c2 = ast.literal_eval(beta)

        dist = BetaEx(
            concentration1=torch.tensor(c1, dtype=torch.float32), concentration0=torch.tensor(c2, dtype=torch.float32))
        return (dist,)
