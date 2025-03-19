from typing import Any, Dict
import ast
import torch
from torch.distributions import Chi2
import scipy.stats as scst

class Chi2Ex(Chi2):
    """
    A class to extend Chi2 to add missing functionality.

    pragma: skip_doc
    """
    def icdf(self, q: torch.Tensor):
        """
        Computes the inverse of cumulative distribution function (CDF) of the Chi-squared distribution.

        Args:
            q (torch.Tensor): Input tensor containing values where the ICDF is evaluated.

        Returns:
            torch.Tensor: Tensor containing the inverse of cumulative probabilities.
        """
        a = q.detach().cpu().numpy()
        df = self.df.detach().cpu().item()
        outputs = scst.chi2.ppf(a, df)
        return torch.tensor(outputs, dtype=torch.float32).to(q.device)


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

        dist = Chi2Ex(
            df=torch.tensor(d, dtype=torch.float32))
        return (dist,)
