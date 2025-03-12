from typing import Any, Dict, Tuple
import torch
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PtoLrSchedulerReduceOnPlateau:
    """
    Creates a reduce-on-plateau learning rate scheduler for an optimizer.

    Parameters  

    **optimizer**  
      (torch.optim.Optimizer) – Wrapped optimizer whose learning rate needs scheduling.

    **grace_period**  
      (int) – Number of epochs to monitor the val loss reduction. If loss does not decrease within grace period epochs, learning rate will be reduced by multiplying the gamma.  E.g. if gamma is set to 0.5, new learning rate will be 0.5 * current learning rate.
       
    **gamma**  
      (float) – Learning rate multiplier to be applied when grace period is up.

    category: PyTorch wrapper - Training
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
                "optimizer": ("PTOPTIMIZER", {}),
                "grace_period": ("INT", {"default": 10, "min": 0, "max": int(1e8)}),
                "gamma": ("FLOAT", {"default": 0.1, "min": 1e-6, "max": 1.0, "step": 1e-8})
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PTLRSCHEDULER",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self, optimizer: torch.optim.Optimizer, grace_period: int, gamma: float) -> Tuple[ReduceLROnPlateau]:
        """
        Creates a reduce-on-plateau learning rate scheduler for an optimizer.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
            grace_period (int): Number of epochs to monitor for val loss reduction.
            gamma (float): Multiplicative factor of learning rate decay.

        Returns:
            tuple: A tuple containing the CosineAnnealingLR scheduler.
        """
        with torch.inference_mode(False):
            lrs = ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=grace_period)
            return (lrs,)
