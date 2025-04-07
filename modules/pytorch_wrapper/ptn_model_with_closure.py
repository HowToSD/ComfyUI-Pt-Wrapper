from typing import Any, Dict, Tuple, Callable, Optional
import torch
import torch.nn as nn


class PtnModelWithClosureDef(nn.Module):
    """
    A model that is followed by a closure function.

    ### Fields:  
    - `model`: The model in the chain.
    - `closure`: Differentiable function to be called at the end of forward.

    pragma: skip_doc
    """

    def __init__(self,
                 model: nn.Module,
                 closure: Callable) -> None:
        super().__init__()
        self.model = model
        self.closure = closure

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        A model that is followed by a closure function.

        Args:  
        - `model`: The model in the chain.
        - `closure`: Differentiable function to be called at the end of forward.

        Returns:
            torch.Tensor: Output tensor after passing through `model` and `closure`.
        """
        x = self.model(inputs)
        x = self.closure(x)
        return x


class PtnModelWithClosure:
    """
    A model that is followed by a closure function.

    ### Fields:  
    - `model`: The model in the chain.
    - `closure`: Differentiable function to be called at the end of forward.    category: PyTorch wrapper - Model
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
                "model": ("PTMODEL", {}),
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
          closure: Callable) -> Tuple[nn.Module]:
        """
        A model that is followed by a closure function.

        Args:  
        - `model`: The model in the chain.
        - `closure`: Differentiable function to be called at the end of forward.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated model.
        """
        with torch.inference_mode(False):
            model = PtnModelWithClosureDef(model, closure)
        return (model,)
