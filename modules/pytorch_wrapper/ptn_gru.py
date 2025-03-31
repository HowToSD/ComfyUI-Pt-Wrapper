from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnGRU:
    """
    Ptn GRU:
    A gated recurrent unit (GRU) model consisting of one or more of a recurrent layer.  

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of output features of the hidden layer matrix.
            num_layers (int): Number of hidden layers.
            bias (bool): Use bias or not.
            batch_first (bool): Indicates the shape of the input. Input is assumed to be a rank-3 tensor. If True, the shape is [Batch, Seq, Token]. If False, the shape is [Seq, Batch, Token]. Note that default is set to `True` unlike PyTorch's GRU default model parameter.
            dropout (float): Dropout probability to be applied to the output of intermediate layers if non-zero. This will not be applied to the last layer's output.
            bidirectional (bool): Processes the input in a backward direction and append the result to the output of the forward direction.

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
                "input_size": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "hidden_size": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "num_layers": ("INT", {"default": 1, "min": 1, "max": 1e3}),
                "bias": ("BOOLEAN", {"default": True}),
                "batch_first": ("BOOLEAN", {"default": True}),
                "dropout": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001}),
                "bidirectional": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          input_size: int,
          hidden_size: int,
          num_layers: int,
          bias: bool,
          batch_first: bool,
          dropout: float,
          bidirectional: bool
        ) -> Tuple[nn.Module]:
        """
        Constructs a GRU model.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of output features of the hidden layer matrix.
            num_layers (int): Number of hidden layers.
            bias (bool): Use bias or not.
            batch_first (bool): Indicates the shape of the input. Input is assumed to be a rank-3 tensor. If True, the shape is [Batch, Seq, Token]. If False, the shape is [Seq, Batch, Token].
            dropout (float): Dropout probability to be applied to the output of intermediate layers if non-zero. This will not be applied to the last layer's output.
            bidirectional (bool): Processes the input in a backward direction and append the result to the output of the forward direction.
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )

            # TODO: Xaviar normal weight initialization is not
            # set up for GRU. If performance turning is needed, consider adding it here.

        return (model,)
