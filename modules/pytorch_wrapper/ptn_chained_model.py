from typing import Any, Dict, Tuple, Callable, Optional
import torch
import torch.nn as nn


class PtnChainedModelDef(nn.Module):
    """
    A chained model that sequentially applies `model_a` and `model_b`,
    optionally followed by a closure function.

    ### Fields:  
    - `model_a`: First model in the chain.
    - `model_b`: Second model in the chain.
    - `closure`: Optional differentiable function to be called at the end of forward.

    Example:
        >>> model_a = nn.Linear(10, 20)
        >>> model_b = nn.Linear(20, 5)
        >>> closure = torch.nn.ReLU()
        >>> model = PtnChainedModelDef(model_a, model_b, closure)
        >>> x = torch.randn(1, 10)
        >>> output = model(x)

    pragma: skip_doc
    """

    def __init__(self,
                 model_a: nn.Module,
                 model_b: nn.Module,
                 closure: Optional[Callable] = None) -> None:
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.closure = closure

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the chained models.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through `model_a`, `model_b`, and optionally `closure`.
        """
        x = self.model_a(inputs)
        x = self.model_b(x)  # Fixed the incorrect use of `inputs` instead of `x`
        if self.closure:
            x = self.closure(x)
        return x


class PtnChainedModel:
    """
    Constructs a chained PyTorch model.

    ### Fields:  
    - model_a: First model in the chain.
    - model_b: Second model in the chain.
    - closure: Optional differentiable function to be called at the end of forward.

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
                "model_a": ("PTMODEL", {}),
                "model_b": ("PTMODEL", {})
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
          model_a: nn.Module,
          model_b: nn.Module,
          closure: Optional[Callable] = None) -> Tuple[nn.Module]:
        """
        Constructs a chained model with optional closure.

        Args:
            model_a (nn.Module): First model in the chain.
            model_b (nn.Module): Second model in the chain.
            closure (Callable, optional): A function applied after `model_b`.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated model.
        """
        with torch.inference_mode(False):
            model = PtnChainedModelDef(model_a, model_b, closure)
        return (model,)
