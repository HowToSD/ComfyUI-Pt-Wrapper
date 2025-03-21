from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnHuberLoss:
    """
    Ptn Huber Loss:

    A class to compute the Huber loss.

    Please see [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html) for details.
    

        Args:  
            reduction (str): Reduction method.  
                              `none`: no reduction will be applied done.   
                              `mean`: computes the mean.  
                              `sum`: computes the sum.  
            delta (float): The threshold used to switch between adjusted MSE and L1 loss.
  
    category: PyTorch wrapper - Loss function
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types for the function.

        Returns:
            Dict[str, Any]: A dictionary specifying required input types.
        """
        return {
            "required": {
                "reduction": (("mean", "sum", "none"),),
                "delta": ("FLOAT", {"default":1.0, "min":0, "max":1e9, "step":0.0000000001}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTLOSS",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          reduction: str,
          delta: float
          ) -> Tuple[nn.Module]:
        """
        A model to compute the Huber loss.

        Args:
            reduction: (str): Reduction method.
                              `none`: no reduction will be applied done.
                              `mean`: computes the mean.
                              `sum`: computes the sum.
            beta (float): The threshold used to switch between adjusted MSE and L1 loss.
            
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = nn.HuberLoss(reduction=reduction, delta=delta)

        return (model,)
