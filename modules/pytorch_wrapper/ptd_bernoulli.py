from typing import Any, Dict
import ast
import torch
from torch.distributions import Bernoulli
import scipy.stats as scst


class BernoulliEx(Bernoulli):
    """
    A class to extend Bernoulli to add missing functionality.

    pragma: skip_doc
    """
    def cdf(self, x: torch.Tensor):
        """
        Computes the cumulative distribution function (CDF) of the Bernoulli distribution.

        Args:
            x (torch.Tensor): Input tensor containing values where the CDF is evaluated.

        Returns:
            torch.Tensor: Tensor containing the cumulative probabilities.
        """
        a = x.detach().cpu().numpy()
        probs = self.probs.detach().cpu().numpy()
        outputs = scst.bernoulli.cdf(a, probs)
        return torch.tensor(outputs, dtype=torch.float32)


class PtdBernoulli:
    """
    Instantiates a Bernoulli distribution object.

        Args:
            probs (str): Probabilities of the distsribution. You can specify a scalar or a tuple of float.  
            logits (str): Logits of the distribution. You can specify a scalar or a tuple of float.  
    
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
                "probs": ("STRING", {"default": ""}),
                "logits": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("PTDISTRIBUTION",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, probs: str, logits: str) -> tuple:
        """
        Instantiates a Bernoulli distribution object.
        
        Args:
            probs (str): Probabilities of the distsribution. You can specify a scalar or a tuple of float.  
            logits (str): Logits of the distribution. You can specify a scalar or a tuple of float.  
        Returns:
            tuple: (torch.distributions.bernoulli.Bernoulli)
        """
        p = ast.literal_eval(probs) if probs else None
        l = ast.literal_eval(logits) if logits else None

        if p is not None and l is not None:  # You cannot use if p because p or l can be "0".
            raise ValueError("You can specify either probabilities or logits, but not both.")
        if p is None and l is None:
            raise ValueError("You have to specify either probabilities or logits")
        
        if p is not None:
            dist = BernoulliEx(probs=torch.tensor(p, dtype=torch.float32))
        elif l is not None:
            dist = BernoulliEx(logits=torch.tensor(l, dtype=torch.float32))

        return (dist,)