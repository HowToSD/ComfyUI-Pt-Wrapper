from typing import Any, Dict
import torch
import ast

class PtdmSample:
    """
    Samples from the input distribution.

    Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.  
            sample_shape (torch.Size): Shape of the sample.
            
    category: PyTorch wrapper - Distribution
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types for the function.

        Returns:
            Dict[str, Any]: A dictionary specifying the required input types.
        """
        return {
            "required": {
                "distribution": ("PTDISTRIBUTION", {}),
                "sample_shape": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES: tuple = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self,
          distribution:torch.distributions.distribution.Distribution,
          sample_shape: str) -> tuple:
        """
        Samples from the input distribution.

        Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.
            sample_shape (torch.Size): Shape of the sample.

        Returns:
            tuple containing the sample
        """

        shape = ast.literal_eval(sample_shape)
        if isinstance(shape, int):
            shape = (shape,)
        return (distribution.sample(shape),)
