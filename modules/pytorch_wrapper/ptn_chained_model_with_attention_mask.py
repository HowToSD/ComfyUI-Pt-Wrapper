from typing import Any, Dict, Tuple, Callable, Optional
import torch
import torch.nn as nn


class PtnChainedModelWithAttentionMaskDef(nn.Module):
    """
    A chained model that sequentially applies `model_a` and `model_b` optionally followed by a closure function. The model also takes attention mask for input.

    ### Fields:  
    - `model_a`: First model in the chain.
    - `model_b`: Second model in the chain.
    - `model_a_mask_req`: True if model_a requires attention_mask for input. For example, Set True if you plug in the `Ptn Multihead Attention` node.
    - `model_b_mask_req`: True if model_b requires attention_mask for input
    - `closure`: Optional differentiable function to be called at the end of forward.

    pragma: skip_doc
    """

    def __init__(self,
                 model_a: nn.Module,
                 model_b: nn.Module,
                 model_a_mask_req: bool,
                 model_b_mask_req: bool,
                 closure: Optional[Callable] = None) -> None:
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.model_a_mask_req = model_a_mask_req
        self.model_b_mask_req = model_b_mask_req
        self.closure = closure

    def forward(self, inputs: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the chained models.

        Args:
            inputs (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor after passing through `model_a`, `model_b`, and optionally `closure`.
        """
        if self.model_a_mask_req:
            x = self.model_a(inputs, mask)
        else:
            x = self.model_a(inputs)

        if self.model_b_mask_req:
            x = self.model_b(x, mask)
        else:
            x = self.model_b(x)
        if self.closure:
            x = self.closure(x)
        return x


class PtnChainedModelWithAttentionMask:
    """
    A chained model that sequentially applies `model_a` and `model_b`,
    optionally followed by a closure function.

    ### Fields:  
    - `model_a`: First model in the chain.
    - `model_b`: Second model in the chain.
    - `model_a_mask_req`: True if model_a requires attention_mask for input
    - `model_b_mask_req`: True if model_b requires attention_mask for input
    - closure: Optional differentiable function to be called at the end of forward.

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
                "model_a": ("PTMODEL", {}),
                "model_b": ("PTMODEL", {}),
                "model_a_mask_req": ("BOOLEAN", {"default": False}),
                "model_b_mask_req": ("BOOLEAN", {"default": False}),
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
          model_a_mask_req: bool,
          model_b_mask_req: bool,
          closure: Optional[Callable] = None) -> Tuple[nn.Module]:
        """
        Constructs a chained model with optional closure.

        Args:
            model_a (nn.Module): First model in the chain.
            model_b (nn.Module): Second model in the chain.
            model_a_mask_req (bool): Whether model_a needs attention mask.
            model_b_mask_req (bool): Whether model_b needs attention mask.
            closure (Callable, optional): A function applied after `model_b`.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated model.
        """
        with torch.inference_mode(False):
            model = PtnChainedModelWithAttentionMaskDef(
                model_a=model_a,
                model_b=model_b,
                model_a_mask_req=model_a_mask_req,
                model_b_mask_req=model_b_mask_req,
                closure=closure
            )
        return (model,)