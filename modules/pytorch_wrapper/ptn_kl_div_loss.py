from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnKLDivLoss:
    """
    Ptn KL Div Loss:
    A class to compute the KL divergence loss.

        Args:  
            reduction (str): Reduction method.  
                              `none`: no reduction will be applied done.   
                              `batchmean`: computes the sum then divides by the input size (recommended in the PyTorch document).
                              `mean`: computes the mean.  
                              `sum`: computes the sum.  

            log_target (bool): Set to `True` if the target is in log probability instead of probability.
                               The input (y_hat) must always be in log-probability form, regardless of this flag.
    
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
                "reduction": (("batchmean", "mean", "sum", "none"),),
                "log_target": ("BOOLEAN", {"default": False})
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
          log_target: bool
          ) -> Tuple[nn.Module]:
        """
        A model to compute the cross entropy loss.

        Args:
            reduction: (str): Reduction method.
                              `none`: no reduction will be applied done.
                              `batchmean`: computes the sum then divides by the input size (recommended in the PyTorch document).
                              `sum`: computes the sum.
                              `mean`: computes the mean.
            log_target (bool): Set to `True` if the target is in log probability instead of probability.
                               The input (y_hat) must always be in log-probability form, regardless of this flag.
                                          
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = nn.KLDivLoss(
                reduction=reduction,
                log_target=log_target)

        return (model,)
