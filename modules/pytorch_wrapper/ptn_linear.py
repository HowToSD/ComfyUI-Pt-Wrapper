from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnLinear:
    """
    A linear model consisting of a single dense layer.  

    ### Fields:  
    - `in_features`: The number of input features.
    - `out_features`: The number of output features.
    - `bias`: Use bias or not.

    category: PyTorch wrapper - Model
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
                "in_features": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "out_features": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "bias": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          in_features: int,
          out_features: int,
          bias: bool
          ) -> Tuple[nn.Module]:
        """
        Constructs a dense model.

        Args:
            in_features (int): The number of input features
            out_features (int): The number of output features
            bias (bool): Use bias or not.
            
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = nn.Linear(in_features, out_features, bias)
            
        return (model,)
