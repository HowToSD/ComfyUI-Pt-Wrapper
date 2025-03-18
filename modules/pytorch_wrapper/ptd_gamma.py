from typing import Any, Dict
import ast
import torch
from torch.distributions import Gamma

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

        dist = Gamma(
            concentration=torch.tensor(c, dtype=torch.float32),
            rate=torch.tensor(r, dtype=torch.float32))
        return (dist,)
