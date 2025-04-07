from typing import Any, Dict, Tuple, Callable, Optional
import torch
import torch.nn as nn


class PtnResidualConnectionModelWithAttentionMaskDef(nn.Module):
    """
    A model that saves the input and add to the output of the specified model.
    optionally followed by a closure function.

    ### Fields:  
    - `model`: A model to process input.
    - `closure`: Optional differentiable function to be called at the end of forward.

    pragma: skip_doc
    """

    def __init__(self,
                 model: nn.Module,
                 closure: Optional[Callable] = None) -> None:
        super().__init__()
        self.model = model
        self.closure = closure

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the chained models.

        Args:
            inputs (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output tensor after passing through `model` and adding residual input followed optionally by `closure`.
        """
        res = inputs
        x = self.model(inputs, mask)
        x = res + x
        if self.closure:
            x = self.closure(x)
        return x


class PtnResidualConnectionModelWithAttentionMask:
    """
    A model that saves the input and add to the output of the specified model.
    optionally followed by a closure function.

    ### Fields:  
    - `model`: A model to process input.
    - `closure`: Optional differentiable function to be called at the end of forward.

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
            },
            "optional": {
                "closure": ("PTCALLABLE", {})
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          model: nn.Module,
          closure: Optional[Callable] = None) -> Tuple[nn.Module]:
        """
        Constructs a chained model with optional closure.

        Args:
            model (nn.Module): A model that saves the input and add to the output of the specified model.
            closure ( Optional[Callable]): An optional differentiable function to be called at the end of forward.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated model.
        """
        with torch.inference_mode(False):
            model = PtnResidualConnectionModelWithAttentionMaskDef(model, closure)
        return (model,)
