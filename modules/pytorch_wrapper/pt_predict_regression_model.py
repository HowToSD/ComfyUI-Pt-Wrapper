from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PtPredictRegressionModel:
    """
    Performs inference on input data.

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
                "model": ("PTMODEL", {}),
                "inputs": ("TENSOR", {}),
                "use_gpu": ("BOOLEAN", {"default":False})
            }
        }

    RETURN_NAMES: Tuple[str] = ("y_hat",)
    RETURN_TYPES: Tuple[str] = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self, model: nn.Module,
          inputs: torch.Tensor,
          use_gpu: bool) -> Tuple[torch.Tensor]:
        """
        Performs inference on test data.

        Args:
            model (torch.nn.Module): The PyTorch model to evaluate.
            inputs (torch.Tensor): Input tensor.
            use_gpu (bool): Whether to use GPU for inference.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the prediction.
        """

        with torch.inference_mode():
            if use_gpu:
                model.to("cuda")
            model.train(False)  # TODO: Change to model.*e*v*a*l() once Comfy's security checker is fixed.
            x = inputs
            if x.dim() == 1:  # Add axis for col
                x = torch.unsqueeze(x, -1)
            if use_gpu:
               x = x.to("cuda")
            outputs = model(x)
            if use_gpu:
                outputs = outputs.to("cpu")
        return (outputs,)
