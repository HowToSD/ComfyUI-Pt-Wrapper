from typing import Any, Dict, Tuple
import ast
import torch
from .ptn_resnet_model_def import ResnetModel


class PtnResnetModel:
    """
    A Resnet model consisting of multiple Resnet layers.  

    ### Fields:  
    - `input_dim`: A string representing the input dimensions in the format "(C,H,W)".  
      Example: `"(3,32,32)"` for a 3-channel 28x28 image.  

    - `output_dim`: An integer specifying the number of output features, which should match the number of target classes in classification.
      Default: `10`, Min: `1`, Max: `1e6`.  

    - `num_blocks`: Number of Resnet blocks for 64 channel blocks. Same number of blocks will be created for 128 channel and 256 channel blocks.

    category: PyTorch wrapper - Training
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
                "input_dim": ("STRING", {"default": "(3,28,28)"}),
                "output_dim": ("INT", {"default": 10, "min": 1, "max": 1e6}),
                "num_blocks": ("INT", {"default": 2, "min": 1, "max": 100})
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self, 
          input_dim:str,
          output_dim: int,
          num_blocks: int) -> None:
        """
        Constructs a Resnet model.

        Args:
            input_dim (str): A string representing the input dimensions in the format "(C,H,W)". Example: `"(3,32,32)"` for a 3-channel 28x28 image.  

            output_dim (int): An integer specifying the number of output features, which should match the number of target classes in classification.
            Default: `10`, Min: `1`, Max: `1e6`.  

            num_blocks (int): Number of Resnet blocks for 64 channel blocks. Same number of blocks will be created for 128 channel and 256 channel blocks.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        input_dim_parsed = tuple(ast.literal_eval(input_dim))

        if len(input_dim_parsed) != 3:
            raise ValueError("Input dimension needs to contain (input channels, input height, input width)")

        if num_blocks < 1:
            raise ValueError("Num_blocks has to be equal to or greater than 1")
        
        if output_dim < 1:
            raise ValueError("Output_dim has to be equal to or greater than 1")

        with torch.inference_mode(False):
            model = ResnetModel(
                        num_blocks=num_blocks,
                        in_channels=input_dim_parsed[0], # in_channels
                        input_height=input_dim_parsed[1],
                        input_width=input_dim_parsed[2],
                        output_dim=output_dim)
            
        return (model,)
