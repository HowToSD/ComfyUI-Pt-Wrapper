from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from .utils import str_to_dim

class PtnInstanceNorm2d:
    """
    Ptn Instance Norm 2d:
    A normalization model to normalize elements over spatial axes within each channel for each sample.

        Args:
            num_features (int): Specifies the size of channel axis.
                - For an image input of shape `[8, 4, 256, 256]`, specify 4.

            affine (bool): If `True`, applies a learnable scaling factor and bias to the normalized elements.

            track_running_stats (bool): If `True`, use exponential moving average (EMA) to store mean and std to be used in eval for normalization. For training, current sample data is always used for mean and std irrespective of this flag.
             
            momentum (float): The EMA coefficient used if track_running_stats is set to True. Higher value gives more weight to the current statistics (mean & std).
             

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
                "num_features": ("INT", {"default":1, "min":1, "max":1e6}),
                "affine": ("BOOLEAN", {"default": True}),
                "track_running_stats": ("BOOLEAN", {"default": True}),
                "momentum": ("FLOAT", {"default":0.1, "min":0, "max":1, "step":1e-6}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          num_features: int,
          affine: bool,
          track_running_stats: bool,
          momentum: float
          ) -> Tuple[nn.Module]:
        """
        A normalization model to normalize elements over spatial axes within each channel for each sample.

        Args:
            num_features (int): Specifies the size of channel axis.
                - For an image input of shape `[8, 4, 256, 256]`, specify 4.

            affine (bool): If `True`, applies a learnable scaling factor and bias to the normalized elements.

            track_running_stats (bool): If `True`, use exponential moving average (EMA) to store mean and std to be used in eval for normalization. For training, current sample data is always used for mean and std irrespective of this flag.
             
            momentum (float): The EMA coefficient used if track_running_stats is set to True. Higher value gives more weight to the current statistics (mean & std).
            
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """

        with torch.inference_mode(False):
            model = nn.InstanceNorm2d(
                num_features=num_features,
                affine=affine,
                track_running_stats=track_running_stats,
                momentum=momentum)
                    
        return (model,)
