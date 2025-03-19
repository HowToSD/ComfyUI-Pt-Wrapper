from typing import Any, Dict
import ast
import torch
from torch.distributions import Binomial
import scipy.stats as scst

class BinomialEx(Binomial):
    """
    A class to extend Binomial to add missing functionality.

    pragma: skip_doc
    """
    def cdf(self, x: torch.Tensor):
        """
        Computes the cumulative distribution function (CDF) of the Binomial distribution.

        Args:
            x (torch.Tensor): Input tensor containing values where the CDF is evaluated.

        Returns:
            torch.Tensor: Tensor containing the cumulative probabilities.
        """
        a = x.detach().cpu().numpy()
        n = self.total_count.detach().cpu().numpy()
        probs = self.probs.detach().cpu().numpy()
        outputs = scst.binom.cdf(a, n, probs)
        return torch.tensor(outputs, dtype=torch.float32).to(x.device)

    def icdf(self, q: torch.Tensor):
        """
        Computes the inverse of cumulative distribution function (CDF) of the Binomial distribution.

        Args:
            q (torch.Tensor): Input tensor containing values where the ICDF is evaluated.

        Returns:
            torch.Tensor: Tensor containing the inverse of cumulative probabilities.
        """
        a = q.detach().cpu().numpy()
        n = self.total_count.detach().cpu().numpy()
        probs = self.probs.detach().cpu().numpy()
        outputs = scst.binom.ppf(a, n, probs)
        return torch.tensor(outputs, dtype=torch.float32).to(q.device)



class PtdBinomial:
    """
    Instantiates a Binomial distribution object.

        Args:
            total_count (str): Total count, or n. You can specify a scalar or a tuple of int.
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
                "total_count": ("STRING", {"default": ""}),
                "probs": ("STRING", {"default": ""}),
                "logits": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("PTDISTRIBUTION",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, total_count: str, probs: str, logits: str):
        """
        Instantiates a Binomial distribution object.
        
        Args:
            total_count (str): Total count, or n. You can specify a scalar or a tuple of int.
            probs (str): Probabilities of the distsribution. You can specify a scalar or a tuple of float.  
            logits (str): Logits of the distribution. You can specify a scalar or a tuple of float.  

        Returns:
            tuple: (torch.distributions.binomial.Binomial)
        """

        n = ast.literal_eval(total_count)
        p = ast.literal_eval(probs) if probs else None
        l = ast.literal_eval(logits) if logits else None
        
        if p is not None and l is not None:  # You cannot use if p because p or l can be "0".
            raise ValueError("You can specify either probabilities or logits, but not both.")
        if p is None and l is None:
            raise ValueError("You have to specify either probabilities or logits")
        
        if p is not None:
            dist = BinomialEx(
                total_count=torch.tensor(n, dtype=torch.int64),
                probs=torch.tensor(p, dtype=torch.float32))
        elif l is not None:
            dist = BinomialEx(
                total_count=torch.tensor(n, dtype=torch.int64),
                logits=torch.tensor(l, dtype=torch.float32))
            
        return(dist,)
