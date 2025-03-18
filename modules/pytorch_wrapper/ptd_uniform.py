from typing import Any, Dict
import ast
import torch
from torch.distributions import Uniform

class PtdUniform:
    """
    Instantiates a Uniform distribution object.

        Args:
            low (str): Minimum (inclusive). You can specify a scalar or a tuple of float.  
            high (str): Maximum (exclusive). You can specify a scalar or a tuple of float.  
    
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
                "low": ("STRING", {"default": ""}),
                "high": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("PTDISTRIBUTION",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, low: str, high: str) -> tuple:
        """
        Instantiates a Uniform distribution object.
        
        Args:
            low (str): Minimum (inclusive). You can specify a scalar or a tuple of float.  
            high (str): Maximum (exclusive). You can specify a scalar or a tuple of float.  

        Returns:
            tuple: (torch.distributions.binomial.Uniform)
        """
        l = ast.literal_eval(low)
        h = ast.literal_eval(high)

        dist = Uniform(
            low=torch.tensor(l, dtype=torch.float32), high=torch.tensor(h, dtype=torch.float32))
        return (dist,)
