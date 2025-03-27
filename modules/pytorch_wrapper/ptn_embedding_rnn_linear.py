from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from pytorch_wrapper.rnn_output_reshape import extract_rnn_return_value_with_adjusted_label


class EmbeddingRNNLinear(nn.Module):
    """
    A recurrent neural network (RNN) with a linear output layer.
    
    This module uses a standard PyTorch RNN, followed by a linear layer applied to the final output tensor.

    pragma: skip_doc
    """

    def __init__(self,
        vocabulary_size: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        nonlinearity: str,
        bias: bool,
        batch_first: bool,
        dropout: float,
        bidirectional: bool,
        linear_output_size: int,
        linear_bias: bool
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            input_size
        )

        self.rnn_model = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        directions = 2 if bidirectional else 1

        # Initialize RNN weights
        for layer in range(num_layers):
            for direction in range(directions):
                suffix = f"_reverse" if direction == 1 else ""
                weight_ih = getattr(self.rnn_model, f"weight_ih_l{layer}{suffix}")
                weight_hh = getattr(self.rnn_model, f"weight_hh_l{layer}{suffix}")

                if nonlinearity == "relu":
                    torch.nn.init.kaiming_normal_(weight_ih, nonlinearity='relu')
                    torch.nn.init.kaiming_normal_(weight_hh, nonlinearity='relu')
                else:
                    torch.nn.init.xavier_normal_(weight_ih)
                    torch.nn.init.xavier_normal_(weight_hh)

        self.linear_model = nn.Linear(
            in_features=hidden_size * directions,
            out_features=linear_output_size,
            bias=linear_bias
        )
        torch.nn.init.xavier_uniform_(self.linear_model.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RNN and the linear layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch, seq, input_size) if batch_first=True.

        Returns:
            torch.Tensor: Output of the linear layer.
        """
        tokens, masks = inputs
        embeds = self.embedding(tokens)
        out_seq, _ = self.rnn_model(embeds)

        x = extract_rnn_return_value_with_adjusted_label(
            masks.to(torch.float32).unsqueeze(-1),  # This is needed to find out padded tokens
            None,
            out_seq,
            self.bidirectional,
            self.batch_first,
            False,
            return_valid_token_mean=True
        )

        outputs = self.linear_model(x)
        return outputs


class PtnEmbeddingRNNLinear:
    """
    Ptn RNN Linear:
    A recurrent neural network (RNN) model with a linear head.
    
        Args:
            vocabulary_size (int): The vocabulary size of the tokens.
            input_size (int): The number of input features.
            hidden_size (int): The number of output features of the hidden layer matrix.
            num_layers (int): Number of hidden layers in RNN.
            nonlinearity (str): Activation function to apply after each hidden layer. Specify 'tanh' or 'relu'.
            bias (bool): Use bias or not.
            batch_first (bool): Indicates the shape of the input. Input is assumed to be a rank-3 tensor. If True, the shape is [Batch, Seq, Token]. If False, the shape is [Seq, Batch, Token]. Note that default is set to `True` unlike PyTorch's RNN default model parameter.
            dropout (float): Dropout probability to be applied to the output of intermediate layers if non-zero. This will not be applied to the last layer's output.
            bidirectional (bool): Processes the input in a backward direction and append the result to the output of the forward direction.
            linear_output_size (int): The number of output features of the linear layer.
            linar_bias (bool): Use bias or not in linear layer.

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
                "vocabulary_size": ("INT", {"default": 10000, "min": 1, "max": 1e6}),
                "input_size": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "hidden_size": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "num_layers": ("INT", {"default": 1, "min": 1, "max": 1e3}),
                "nonlinearity": (("tanh", "relu"),),
                "bias": ("BOOLEAN", {"default": True}),
                "batch_first": ("BOOLEAN", {"default": True}),
                "dropout": ("FLOAT", {"default": 0, "min": 0, "max": 1}),
                "bidirectional": ("BOOLEAN", {"default": False}),
                "linear_output_size": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "linear_bias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          vocabulary_size: int,
          input_size: int,
          hidden_size: int,
          num_layers: int,
          nonlinearity: str,
          bias: bool,
          batch_first: bool,
          dropout: float,
          bidirectional: bool,
          linear_output_size: int,
          linear_bias: bool
        ) -> Tuple[nn.Module]:
        """
        Constructs an RNN model.

        Args:
            vocabulary_size (int): The vocabulary size of the tokens.
            input_size (int): The number of input features.
            hidden_size (int): The number of output features of the hidden layer matrix.
            num_layers (int): Number of hidden layers.
            nonlinearity (str): Activation function to apply after each hidden layer. Specify 'tanh' or 'relu'.
            bias (bool): Use bias or not.
            batch_first (bool): Indicates the shape of the input. Input is assumed to be a rank-3 tensor. If True, the shape is [Batch, Seq, Token]. If False, the shape is [Seq, Batch, Token].
            dropout (float): Dropout probability to be applied to the output of intermediate layers if non-zero. This will not be applied to the last layer's output.
            bidirectional (bool): Processes the input in a backward direction and append the result to the output of the forward direction.
            linear_output_size (int): The number of output features of the linear layer.
            linar_bias (bool): Use bias or not in linear layer.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = EmbeddingRNNLinear(
                vocabulary_size=vocabulary_size,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
                linear_output_size=linear_output_size,
                linear_bias=linear_bias
            )
            return (model,)
