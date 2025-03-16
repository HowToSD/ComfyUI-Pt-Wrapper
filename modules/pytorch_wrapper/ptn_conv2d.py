from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import ast
from .utils import str_to_dim

class PtnConv2d:
    """
    A convolutional model consisting of a single conv2d layer.  

        Args:
            in_channels (int): The number of input channels  
            out_channels (int): The number of output channels  
            kernel_size (str): The size of the kernel  
            stride (str): The stride to be used for sliding the kernel  
            padding (str): The amount of padding added to the input  
            dilation (str): The distance between adjacent elements in the kernel to adjust receptive fields.  
            groups (str): The number of groups used to divide input and output channels for separate processing.  
            bias (bool): If `True`, adds a learnable bias to the output  
        padding_mode (str): Specifies how padding values are set in the input before convolution.  
t. 
            bias (bool): Use bias or not.

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
                "in_channels": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "out_channels": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "kernel_size": ("STRING", {"default":"3", "multiline": False}),
                "stride": ("STRING", {"default":"1", "multiline": False}),
                "padding": ("STRING", {"default":"same", "multiline": False}),
                "dilation": ("STRING", {"default":"1", "multiline": False}),
                "groups": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "bias": ("BOOLEAN", {"default": True}),
                "padding_mode": (("zeros", "reflect", "replicate", "circular"),)
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          in_channels: int,
          out_channels: int,
          kernel_size: str,
          stride: str,
          padding: str,
          dilation: str,
          groups: str,
          bias: bool,
          padding_mode: str
          ) -> Tuple[nn.Module]:
        """
        Constructs a dense model.

        Args:
            in_channels (int): The number of input channels  
            out_channels (int): The number of output channels  
            kernel_size (str): The size of the kernel  
            stride (str): The stride to be used for sliding the kernel
            padding (str): The amount of padding added to the input  
            dilation (str): The distance between adjacent elements in the kernel to adjust receptive fields.
            groups (str): The number of groups used to divide input and output channels for separate processing.  
            bias (bool): If `True`, adds a learnable bias to the output  
        padding_mode (str): Specifies how padding values are set in the input before convolution.  
t. 
            bias (bool): Use bias or not.
            
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        k = str_to_dim(kernel_size)
        s = str_to_dim(stride)
        if(padding == "same"):
            p = padding
        elif(padding=="valid"):
            p = padding
        else:
            p = str_to_dim(padding)
        d = str_to_dim(dilation)

        with torch.inference_mode(False):
            model = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode)
            
        return (model,)
