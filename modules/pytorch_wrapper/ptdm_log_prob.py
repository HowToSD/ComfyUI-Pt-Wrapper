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
                "value": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self,
          distribution:torch.distributions.distribution.Distribution,
          value: str) -> tuple:
        """
        Computes the probability for the input distribution.

        Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.
            value (Union[int,float])): Value or number of successes.

        Returns:
            tuple containing the log of probability
        """

        k = ast.literal_eval(value)
        if isinstance(k, int) is False and isinstance(k, float) is False :
            if isinstance(distribution, Dirichlet) is False:
                raise ValueError("You need to specify a float or an int.")

        if isinstance(distribution, (Poisson, Categorical, Multinomial, Binomial, NegativeBinomial)):
            tens = torch.tensor(k, dtype=torch.int64)
        else:
            tens = torch.tensor(k, dtype=torch.float32)

        return (distribution.log_prob(tens),)
    