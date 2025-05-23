from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from .utils import str_to_dim

class PtnAvgPool2d:
    """
    Ptn Avg Pool 2d:
    An avgpool layer.  

        Args:
            kernel_size (str): The size of the kernel  
            stride (str): The stride to be used for sliding the kernel
            padding (str): The amount of padding added to the input  

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
                "kernel_size": ("STRING", {"default":"2"}),
                "stride": ("STRING", {"default":"2"}),
                "padding": ("STRING", {"default":"0"})
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          kernel_size: str,
          stride: str,
          padding: str
          ) -> Tuple[nn.Module]:
        """
        Constructs a dense model.

        Args:
            kernel_size (str): The size of the kernel  
            stride (str): The stride to be used for sliding the kernel
            padding (str): The amount of padding added to the input  
            
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        k = str_to_dim(kernel_size)
        s = str_to_dim(stride)
        p = str_to_dim(padding)

        with torch.inference_mode(False):
            model = nn.AvgPool2d(
                kernel_size=k,
                stride=s,
                padding=p)
            
        return (model,)
