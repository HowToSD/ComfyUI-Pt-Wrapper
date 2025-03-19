from typing import Any, Dict
import ast
import torch
from torch.distributions import StudentT
import scipy.stats as scst

class StudentTEx(StudentT):
    """
    A class to extend StudentT to add missing functionality.

    pragma: skip_doc
    """
    def cdf(self, x: torch.Tensor):
        """
        Computes the cumulative distribution function (CDF) of the StudentT distribution.

        Args:
            x (torch.Tensor): Input tensor containing values where the CDF is evaluated.

        Returns:
            torch.Tensor: Tensor containing the cumulative probabilities.
        """
        a = x.detach().cpu().numpy()
        df = self.df.detach().cpu().item()
        loc = self.loc.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        outputs = scst.t.cdf((a - loc)/scale, df)
        return torch.tensor(outputs, dtype=torch.float32).to(x.device)


class PtdStudentT:
    """
    Instantiates a StudentT distribution object.

        Args:
            df (str): Degree of freedom. You can specify a scalar or a tuple of float.  
            loc (str): Rate. Mean of the distsribution. You can specify a scalar or a tuple of float.  
            scale (str): Rate. Scale of the distribution. You can specify a scalar or a tuple of float.  
    
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
                "df": ("STRING", {"default": ""}),
                "loc": ("STRING", {"default": ""}),
                "scale": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("PTDISTRIBUTION",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self,
          df: str,
          loc: str,
          scale: str) -> tuple:
        """
        Instantiates a StudentT distribution object.
        
        Args:
            df (str): Degree of freedom. You can specify a scalar or a tuple of float.  
            loc (str): Rate. Mean of the distsribution. You can specify a scalar or a tuple of float.  
            scale (str): Rate. Standard deviation of the distribution. You can specify a scalar or a tuple of float.  

        Returns:
            tuple: (torch.distributions.studentT.StudentT)
        """
        d = ast.literal_eval(df)
        mu = ast.literal_eval(loc)
        s = ast.literal_eval(scale)

        dist = StudentTEx(
            df=torch.tensor(d, dtype=torch.float32),
            loc=torch.tensor(mu, dtype=torch.float32),
            scale=torch.tensor(s, dtype=torch.float32))
        return (dist,)
