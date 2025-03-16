from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnPreFlattenDef(nn.Module):
    """
    Flattens the input tensor before processing the tensor in the specified model.

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
        x = torch.flatten(inputs, start_dim=1)  # Flatten except batch-axis
        x = self.model(x)
        return x


class PtnPreFlatten:
    """
    Flattens the input tensor before processing the tensor in the specified model.

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
            output_model = PtnPreFlattenDef(model)
        return (output_model,)
