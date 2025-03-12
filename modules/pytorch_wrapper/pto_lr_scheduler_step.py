from typing import Any, Dict, Tuple
import torch
import torch.optim
from torch.optim.lr_scheduler import StepLR

class PtoLrSchedulerStep:
    """
    Creates a StepLR learning rate scheduler for an optimizer.

    Parameters  
    
    **optimizer**  
      (torch.optim.Optimizer) – Wrapped optimizer whose learning rate needs scheduling.

    **step_size**  
      (int) – Period of learning rate decay.

    **gamma**  
      (float) – Multiplicative factor of learning rate decay.

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
                "step_size": ("INT", {"default": 10, "min": 1, "max": int(1e8)}),
                "gamma": ("FLOAT", {"default": 0.1, "min": 1e-6, "max": 1.0, "step": 1e-8}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PTLRSCHEDULER",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self, optimizer: torch.optim.Optimizer, step_size: int, gamma: float) -> Tuple[StepLR]:
        """
        Creates a StepLR learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.

        Returns:
            tuple: A tuple containing the StepLR scheduler.
        """
        with torch.inference_mode(False):
            lrs = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return (lrs,)
