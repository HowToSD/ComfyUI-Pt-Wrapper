from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from .utils import str_to_dim

class PtnLayerNorm:
    """
    A normalization model to normalize elements over specified axes.

        Args:
            normalized_shape (list): Specifies the shape of elements to normalize.  
                - For an input of shape `[8, 1024, 768]` (where axes represent `[batch_size, sequence_length, hidden_dim]`):
                - `normalize_shape=[768]` normalizes within each token (across the hidden dimension).
                - `normalize_shape=[1024, 768]` normalizes within each sample (across both sequence and token dimensions).  
                - For an image input of shape `[8, 4, 256, 256]` (where axes represent `[batch_size, channel, height, width]`):
                - `normalize_shape=[256, 256]` normalizes within each channel (across spatial dimensions).
                - `normalize_shape=[4, 256, 256]` normalizes within each sample (across channels and spatial dimensions).

            elementwise_affine (bool): If `True`, applies a learnable scaling factor to the normalized elements.

            bias (bool): If `True` and elementwise_affine is also `True`, applies a learnable bias to the normalized elements.

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
                "normalized_shape": ("STRING", {"default": "[Specify shape here]"}),
                "elementwise_affine": ("BOOLEAN", {"default": True}),
                "bias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          normalized_shape: str,
          elementwise_affine: bool,
          bias: bool
          ) -> Tuple[nn.Module]:
        """
        A normalization model to normalize elements over specified axes.

        Args:
            normalized_shape (list): Specifies the shape of elements to normalize.  
                - For an input of shape `[8, 1024, 768]` (where axes represent `[batch_size, sequence_length, hidden_dim]`):
                - `normalize_shape=[768]` normalizes within each token (across the hidden dimension).
                - `normalize_shape=[1024, 768]` normalizes within each sample (across both sequence and token dimensions).  
                - For an image input of shape `[8, 4, 256, 256]` (where axes represent `[batch_size, channel, height, width]`):
                - `normalize_shape=[256, 256]` normalizes within each channel (across spatial dimensions).
                - `normalize_shape=[4, 256, 256]` normalizes within each sample (across channels and spatial dimensions).

            elementwise_affine (bool): If `True`, applies a learnable scaling factor to the normalized elements.

            bias (bool): If `True` and elementwise_affine is also `True`, applies a learnable bias to the normalized elements.
            
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        shape = str_to_dim(normalized_shape)

        with torch.inference_mode(False):
            model = nn.LayerNorm(
                normalized_shape=shape,
                elementwise_affine=elementwise_affine,
                bias=bias)
            
        return (model,)
