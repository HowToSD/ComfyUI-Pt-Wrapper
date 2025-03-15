from typing import Any, Dict, Union, Iterable
import torch
import torch.optim


class SimpleOptimizer(torch.optim.Optimizer):
    """
    Implementation of the most basic optimizer to update W using the below formula:

$$
W -= lr * G
$$

    In Pytorch,
$$
param.data -= lr * param.grad
$$

but accessing param.data directly can be tricky, so we can use:
$$
param -= lr * param.grad
$$

    step() is used instead of forward() so that LR scheduler can call.

    pragma: skip_doc
    """
    def __init__(self,
                 parameters:Iterable[Union[torch.nn.Parameter, Dict[str, Any]]],
                 lr: float):
        additional_dict = {"lr": lr}
        super().__init__(parameters, additional_dict)

    def step(self):
        with torch.no_grad():
            # Normally, parameter_group is a list containing a single dict.
            # However, if the user passes a dict to have separater parameter
            # for each layer, it can have multiple dicts inside.
            # param_groups is set up in the super constructor.
            # You do not need to define this yourself.
            for g in self.param_groups:  
                for parameter in g["params"]:
                    if hasattr(parameter, "grad") and parameter.grad is not None:
                        parameter -= g["lr"] * parameter.grad


class PtoSimple:
    """
    Pto Simple:
    Instantiates the most basic optimizer to update W using the below formula:
$$
W -= lr * G
$$
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
                "learning_rate": ("FLOAT", {"default":0.001, "min":1e-10, "max":1, "step":0.0000000001}),
            }
        }

    RETURN_TYPES: tuple = ("PTOPTIMIZER",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self, model, learning_rate) -> tuple:
        """
        Instantiates the optimizer.

        Args:
            model (torch.nn.Module): PyTorch model
            learning_rate (float): Learning rate

        Returns:
            tuple: A tuple containing the ndarray.
        """
        with torch.inference_mode(False):
            opt = SimpleOptimizer(model.parameters(), lr=learning_rate)
            return(opt,)