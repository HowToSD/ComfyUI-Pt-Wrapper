from typing import Any, Dict
import ast
import torch
from torch.distributions import Chi2

class PtdChi2:
    """
    Ptd Chi2:
    Instantiates a Chi-squared distribution object.

        Args:
            df (str): Degree of freedom. You can specify a scalar or a tuple of float.  
    
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
                "df": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("PTDISTRIBUTION",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self,
          df: str) -> tuple:
        """
        Instantiates a Chi-squared distribution object.

        Args:
            df (str): Degree of freedom. You can specify a scalar or a tuple of float.  

        Returns:
            tuple: (torch.distributions.binomial.Chi2)
        """
        d = ast.literal_eval(df)

        dist = Chi2(
            df=torch.tensor(d, dtype=torch.float32))
        return (dist,)
