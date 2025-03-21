from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnL1Loss:
    """
    Ptn L1 Loss:
    A class to compute the L1 loss.

        Args:  
            reduction: (str): Reduction method.  
                              `none`: no reduction will be applied done.   
                              `mean`: computes the mean.  
                              `sum`: computes the sum.  


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
                "reduction": (("mean", "sum", "none"),)
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTLOSS",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          reduction: str
          ) -> Tuple[nn.Module]:
        """
        A model to compute the L1 loss.

        Args:
            reduction: (str): Reduction method.
                              `none`: no reduction will be applied done.
                              `mean`: computes the mean.
                              `sum`: computes the sum.

            
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = nn.L1Loss(reduction=reduction)

        return (model,)
