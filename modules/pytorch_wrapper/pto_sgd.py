from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class PtoSGD:
    """
    Pto SGD:
    Instantiates the SGD optimizer.

    Parameters  
            model (PTMODEL): The PyTorch model whose parameters will be optimized.  
            learning_rate (float): The learning rate for the AdamW optimizer.  
            momentum (float):  Coefficient to apply to the past gradient to adjust the contribution of past gradients  
            dampening (float): Coefficient to adjust the contribution of current gradient.  
            weight_decay (float): The weight decay parameter.  
            nesterov (bool): If True, uses the Nesterov version.  

    category: PyTorch wrapper - Optimizer
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
                "learning_rate": ("FLOAT", {"default":0.001, "min":1e-10, "max":1, "step":0.0000000001}),
                "momentum": ("FLOAT", {"default":0.0, "min":0.0, "max":1.0, "step":0.0000000001}),
                "dampening": ("FLOAT", {"default":0.0, "min":0.0, "max":1.0, "step":0.0000000001}),
                "weight_decay": ("FLOAT", {"default":0.0, "min":0.0, "max":1.0, "step":0.0000000001}),
                "nesterov": ("BOOLEAN", {"default":False})
            }
        }

    RETURN_TYPES: tuple = ("PTOPTIMIZER",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self,
          model:torch.nn.Module,
          learning_rate:float,
          momentum:float,
          dampening:float,
          weight_decay:float,
          nesterov: bool) -> tuple:
        """
        Instantiates the optimizer.

        Refer to https://pytorch.org/docs/stable/generated/torch.optim.SGD.html for further details.

        Args:
            model (torch.nn.Module): The PyTorch model whose parameters will be optimized.
            learning_rate (float): The learning rate for the AdamW optimizer.
            momentum (float):  Coefficient to apply to the past gradient to adjust the contribution of past gradients
            dampening (float): Coefficient to adjust the contribution of current gradient.
            weight_decay (float): The weight decay parameter.
            nesterov (bool): If True, uses the Nesterov version.
        Returns:
            tuple: A tuple containing the optimizer.
        """
        with torch.inference_mode(False):
            opt = torch.optim.SGD(model.parameters(),
                                  lr=learning_rate,
                                  momentum=momentum,
                                  dampening=dampening,
                                  weight_decay=weight_decay,
                                  nesterov=nesterov)
            return(opt,)