from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

class PtComputeLoss:
    """
    Computes loss for the input with the target using the specified loss function.
    
    The user does not normally need to use this node. Instead the user should use training nodes that accept a loss object.
    
    category: PyTorch wrapper - Training
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the expected input types.

        Returns:
            Dict[str, Any]: A dictionary specifying required and optional input types.
        """
        return {
            "required": {
                "input_tens": ("TENSOR", {}),
                "target_tens": ("TENSOR", {}),
                "loss": ("PTLOSS", {})
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self,
          input_tens: torch.Tensor,
          target_tens: torch.Tensor,
          loss: nn.Module) -> Tuple[torch.Tensor]:
        """
        Computes loss for the input with the target using the specified loss function.

        Args:
            input_tens (torch.Tensor): Tensor containing the model output or y_hat.
            target_tens (torch.Tensor): Tensor containing the ground truth or the label.
            loss (nn.Module): Loss object instance.

        Returns:
            Tuple[nn.Module]: A tuple containing the computed loss.
        """
        return (loss(input_tens, target_tens),)
