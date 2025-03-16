from typing import Any, Dict, Tuple, Callable, Optional
import torch
import torch.nn as nn


class PtnPreAddChannelAxisDef(nn.Module):
    """
    Adds a channel axis after the batch axis if the input is rank 3 (bs, h, w).

    pragma: skip_doc
    """

    def __init__(self,
                 model: nn.Module
    ):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the `model`.
        """
        if inputs.dim() not in [3, 4]:
            raise ValueError("Inputs is not a rank 3 or 4 tensor.")
        
        if inputs.dim() == 3:
            x = torch.unsqueeze(inputs, 1)
        else:
            x = inputs

        x = self.model(x)
        return x


class PtnPreAddChannelAxis:
    """
    Adds a channel axis after the batch axis if the input is rank 3 (bs, h, w)
    before the forward pass by the specified model.

    category: PyTorch wrapper - Model
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
                "model": ("PTMODEL", {})
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          model: nn.Module):
        """
        Adds a channel axis after the batch axis if the input is rank 3 (bs, h, w).

        Args:
            model (nn.Module): The model.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated model.
        """
        with torch.inference_mode(False):
            output_model = PtnPreAddChannelAxisDef(model)
        return (output_model,)
