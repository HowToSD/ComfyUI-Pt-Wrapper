from typing import Any, Dict, Tuple
import torch
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class PtoLrSchedulerCosineAnnealing:
    """
    Creates a cosine annealing learning rate scheduler for an optimizer.

    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lists:

    eta_t = eta_min + (1/2) * (eta_max - eta_min) * (1 + cos((t_cur / t_max) * pi))

    eta_min is set to 0 by default, so the equation simplifies to:

    eta_t = (1/2) * eta_max * (1 + cos((t_cur / t_max) * pi))

    When t_cur = 0:
      (1 + cos((t_cur / t_max) * pi)) evaluates to (1 + cos(0 * pi)) = (1 + 1) = 2

    When t_cur = t_max:
      (1 + cos((t_cur / t_max) * pi)) evaluates to (1 + cos(1 * pi)) = (1 - 1) = 0

    Therefore, eta_t value changes as follows:
    * Starts from (1/2) * eta_max * 2, which simplifies to eta_max
    * Ends at 0

    Parameters  

    **optimizer**  
      (torch.optim.Optimizer) – Wrapped optimizer whose learning rate needs scheduling.

    **num_epochs**  
      (int) – Number of epochs. This maps to the T_max parameter for CosineAnnealingLR internally.

    **minimum_lr**  
      (float) – Minimum learning rate.

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
                "num_epochs": ("INT", {"default": 10, "min": 1, "max": int(1e8)}),
                "minimum_lr": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 1e-9}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PTLRSCHEDULER",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self, optimizer: torch.optim.Optimizer, num_epochs: int, minimum_lr: float) -> Tuple[CosineAnnealingLR]:
        """
        Creates a cosine annealing learning rate scheduler for an optimizer.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
            num_epochs (int): Number of epochs.
            gamma (float): Minimum learning rate.

        Returns:
            tuple: A tuple containing the CosineAnnealingLR scheduler.
        """
        with torch.inference_mode(False):
            lrs = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=minimum_lr)
            return (lrs,)
