from typing import Any, Dict
import ast
import torch
from torch.distributions import Normal

class PtdNormal:
    """
    Instantiates a Normal distribution object.

        Args:
            loc: Mean of the distsribution. You can specify a scalar or a tuple of float.  
            scale: Standard deviation of the distribution. You can specify a scalar or a tuple of float.  
    
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
                "loc": ("STRING", {"default": ""}),
                "scale": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("PTDISTRIBUTION",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, loc: str, scale: str) -> tuple:
        """
        Instantiates a Normal distribution object.
        
        Args:
            loc (str): Mean of the distsribution. You can specify a scalar or a tuple of float.  
            scale (str): Standard deviation of the distribution. You can specify a scalar or a tuple of float.  

        Returns:
            tuple: (torch.distributions.binomial.Normal)
        """
        mu = ast.literal_eval(loc)
        s = ast.literal_eval(scale)

        dist = Normal(
            loc=torch.tensor(mu, dtype=torch.float32),
            scale=torch.tensor(s, dtype=torch.float32))
        return (dist,)
