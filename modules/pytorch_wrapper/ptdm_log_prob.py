from typing import Any, Dict
import ast
import torch
from torch.distributions import *

class PtdmLogProb:
    """
    Computes the log of probability for the input distribution.

    Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.  
            value (Union[int,float]): Value (Number of successes for PMF).  

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

    def f(self,
          distribution:torch.distributions.distribution.Distribution,
          x: str) -> tuple:
        """
        Computes the probability for the input distribution.

        Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.
            x (Union[int,float])): Value or number of successes.

        Returns:
            tuple containing the log of probability
        """

        v = ast.literal_eval(x)
        if isinstance(v, int) is False and isinstance(v, float) is False :
            if isinstance(distribution, Dirichlet) is False:
                raise ValueError("You need to specify a float or an int.")

        if isinstance(distribution, (Poisson, Categorical, Multinomial, Binomial, NegativeBinomial)):
            tens = torch.tensor(v, dtype=torch.int64)
        else:
            tens = torch.tensor(v, dtype=torch.float32)

        return (distribution.log_prob(tens),)
    