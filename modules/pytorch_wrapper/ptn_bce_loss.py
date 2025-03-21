from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnBCELoss:
    """
    Ptn BCE Loss:
    A class to compute the binary cross entropy loss.

    Unlike the Ptn BCE Loss With Logits node, which takes logits as input, this node expects probabilities. If the input consists of logits, apply a Sigmoid function first or use the Ptn BCE Loss With Logits node instead.

        Args:  
            reduction (str): Reduction method.  
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
        A model to compute the cross entropy loss.

        Args:
            reduction: (str): Reduction method.
                              `none`: no reduction will be applied done.
                              `sum`: computes the sum.
                              `mean`: computes the mean.
            
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = nn.BCELoss(reduction=reduction)

        return (model,)
