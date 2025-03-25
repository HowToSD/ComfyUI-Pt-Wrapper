from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnRNN:
    """
    Ptn RNN:
    A recurrent neural network (RNN) model consisting of one or more of a recurrent layer.  

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of output features of the hidden layer matrix.
            num_layers (int): Number of hidden layers.
            nonlinearity (str): Activation function to apply after each hidden layer. Specify 'tanh' or 'relu'.
            bias (bool): Use bias or not.
            batch_first (bool): Indicates the shape of the input. Input is assumed to be a rank-3 tensor. If True, the shape is [Batch, Seq, Token]. If False, the shape is [Seq, Batch, Token]. Note that default is set to `True` unlike PyTorch's RNN default model parameter.
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
                "nonlinearity": (("tanh", "relu"),),
                "bias": ("BOOLEAN", {"default": True}),
                "batch_first": ("BOOLEAN", {"default": True}),
                "dropout": ("FLOAT", {"default": 0, "min": 0, "max": 1}),
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
          nonlinearity: str,
          bias: bool,
          batch_first: bool,
          dropout: float,
          bidirectional: bool
        ) -> Tuple[nn.Module]:
        """
        Constructs an RNN model.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of output features of the hidden layer matrix.
            num_layers (int): Number of hidden layers.
            nonlinearity (str): Activation function to apply after each hidden layer. Specify 'tanh' or 'relu'.
            bias (bool): Use bias or not.
            batch_first (bool): Indicates the shape of the input. Input is assumed to be a rank-3 tensor. If True, the shape is [Batch, Seq, Token]. If False, the shape is [Seq, Batch, Token].
            dropout (float): Dropout probability to be applied to the output of intermediate layers if non-zero. This will not be applied to the last layer's output.
            bidirectional (bool): Processes the input in a backward direction and append the result to the output of the forward direction.
        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )

            directions = 2 if bidirectional else 1

            # Initialize weights
            for layer in range(num_layers):
                for direction in range(directions):
                    suffix = f"_reverse" if direction == 1 else ""
                    
                    weight_ih = getattr(model, f"weight_ih_l{layer}{suffix}")
                    weight_hh = getattr(model, f"weight_hh_l{layer}{suffix}")

                    # ih for X, hh is for prev H to current H
                    if nonlinearity == "relu":
                        torch.nn.init.kaiming_normal_(weight_ih, nonlinearity='relu')
                        torch.nn.init.kaiming_normal_(weight_hh, nonlinearity='relu')
                    else:
                        torch.nn.init.xavier_normal_(weight_ih)
                        torch.nn.init.xavier_normal_(weight_hh)

                    # bias_ih = getattr(model, f"bias_ih_l{layer}{suffix}")
                    # bias_hh = getattr(model, f"bias_hh_l{layer}{suffix}")

                    # torch.nn.init.uniform_(weight_ih, -0.05, 0.05)
                    # torch.nn.init.uniform_(weight_hh, -0.05, 0.05)
                    # nn.init.zeros_(bias_ih)
                    # nn.init.zeros_(bias_hh)

        return (model,)
