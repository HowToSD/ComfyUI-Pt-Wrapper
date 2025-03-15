from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class PtoAdamW:
    """
    Pto AdamW:
    Instantiates the AdamW optimizer.

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
                "beta1": ("FLOAT", {"default":0.9, "min":1e-10, "max":1, "step":0.0000000001}),
                "beta2": ("FLOAT", {"default":0.999, "min":1e-10, "max":1, "step":0.0000000001}),
                "weight_decay": ("FLOAT", {"default":0.01, "min":0.0, "max":1.0, "step":0.0000000001}),
                "amsgrad": ("BOOLEAN", {"default":False})
            }
        }

    RETURN_TYPES: tuple = ("PTOPTIMIZER",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self,
          model:torch.nn.Module,
          learning_rate:float,
          beta1:float,
          beta2:float,
          weight_decay:float,
          amsgrad: bool) -> tuple:
        """
        Instantiates the optimizer.

        Args:
            model (torch.nn.Module): The PyTorch model whose parameters will be optimized.
            learning_rate (float): The learning rate for the AdamW optimizer.
            beta1 (float): Exponential decay rate for the first moment estimates (moving average of gradients).
            beta2 (float): Exponential decay rate for the second moment estimates (moving average of squared gradients).
            weight_decay (float): The weight decay parameter.
            amsgrad (bool): If True, uses the AMSGrad version.
        Returns:
            tuple: A tuple containing the optimizer.
        """
        with torch.inference_mode(False):
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=learning_rate, betas=(beta1, beta2),
                                    weight_decay=weight_decay,
                                    amsgrad=amsgrad)
            return(opt,)