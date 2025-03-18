from typing import Any, Dict
import ast
import torch
from torch.distributions import Beta

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

        dist = Beta(
            concentration1=torch.tensor(c1, dtype=torch.float32), concentration0=torch.tensor(c2, dtype=torch.float32))
        return (dist,)
