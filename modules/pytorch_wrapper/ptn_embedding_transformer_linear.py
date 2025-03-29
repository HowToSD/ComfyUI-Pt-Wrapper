from typing import Any, Dict, Tuple, Union, Callable, Optional
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from typing import Tuple, Union, Callable


class EmbeddingTransformerLinear(nn.Module):
    """
    A Transformer encoder-based classification model with a linear output layer.

    This module uses a standard PyTorch TransformerEncoder, followed by masked mean pooling
    and a linear layer applied to the pooled representation.

        Args:
            num_layers (int): Number of encoder layers.  
            vocabulary_size (int): Vocabulary size of the tokens.  
            hidden_size (int): Hidden size (d_model) of the Transformer.  
            nhead (int): Number of attention heads.  
            dim_feedforward (int): Feedforward dimension.  
            dropout (float): Dropout rate.  
            nonlinearity (str): Activation function ('relu' or 'gelu').  
            layer_norm_eps (float): Epsilon for layer normalization.  
            batch_first (bool): Input shape format: True for (B, Seq, Feature), False for (Seq, B, Feature).  
            norm_first (bool): Whether normalization comes before sublayers.  
            bias (bool): Use bias in Transformer layers.  
            max_length (int): The maximum sequence length used for allocating the positional  embedding.  This value is not passed to the Transformer itself.  
            linear_output_size (int): Size of linear layer output.  
            linear_bias (bool): Use bias in linear layer.  

    pragma: skip_doc
    """

    def __init__(self,
        num_encoder_layers: int,
        vocabulary_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]],
        layer_norm_eps: float,
        batch_first: bool,
        norm_first: bool,
        bias: bool,
        max_length: int,
        linear_output_size: int,
        linear_bias: bool
    ):
        super().__init__()
        self.batch_first = batch_first

        self.word_embedding = nn.Embedding(vocabulary_size, d_model)
        self.positional_embedding = nn.Embedding(max_length, d_model)

        # FIX: Allocate position_ids buffer once to avoid reallocation in each forward pass
        # self.register_buffer("position_ids_buffer", torch.arange(max_length))  # (max_length,)

        # FIX: use preallocated position_ids_buffer
        # position_ids = self.position_ids_buffer[:seq_len].unsqueeze(0).expand(batch_size, -1)  # (B, T)
        self.position_ids = torch.arange(max_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        self.linear_model = nn.Linear(
            in_features=d_model,
            out_features=linear_output_size,
            bias=linear_bias
        )
        torch.nn.init.xavier_uniform_(self.linear_model.weight)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the Transformer encoder and linear layer.

        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]):
                tokens: (B, T) token IDs,
                masks:  (B, T) attention mask (1 = keep, 0 = pad)

        Returns:
            torch.Tensor: Output of the linear layer (B, linear_output_size)
        """
        tokens, masks = inputs  # (B, Tokens), (B, Tokens)
        if tokens.dim() != 2:
            raise ValueError(f"tokens with rank {tokens.dim()} was passed.")
        if masks.dim() != 2:
            raise ValueError(f"masks with rank {tokens.dim()} was passed.")
        if tokens.size() != masks.size():
            raise ValueError(f"tokens size masks size do not match.")

        batch_size, seq_len = tokens.size()
        token_embeds = self.word_embedding(tokens)

        position_ids_repeated = self.position_ids[:seq_len].repeat(batch_size, 1).to(token_embeds.device)
        positional_embeds = self.positional_embedding(position_ids_repeated)

        if token_embeds.size() != positional_embeds.size():
            raise ValueError("token_embeds & positional_embeds size mismatch.")
        x = token_embeds + positional_embeds  # (B, Seq, Feature)

        if not self.batch_first:
            x = x.transpose(0, 1)  # (Seq, B, Feature)

        x = self.encoder(
            x,
            src_key_padding_mask=~masks.bool()  # (B, Seq)
        )

        if not self.batch_first:
            x = x.transpose(0, 1)  # (B, Seq, Feature)

        # FIX: More memory-efficient masked mean pooling
        mask = masks.float()  # (B, Seq)
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)  # (B, Feature)

        outputs = self.linear_model(x)  # (B, linear_output_size)
        return outputs
    

class PtnEmbeddingTransformerLinear:
    """
    A Transformer encoder model with a linear head.

        Args:
            num_layers (int): Number of encoder layers in Transformer.
            vocabulary_size (int): The vocabulary size of the tokens.
            hidden_size (int): Hidden dimension (d_model) for each token of the Transformer.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Feedforward dimension.
            dropout (float): Dropout rate for Transformer layers.
            nonlinearity (str): Activation function ('relu' or 'gelu') used in Transformer layers.
            layer_norm_eps (float): Epsilon value for layer normalization.
            batch_first (bool): Input shape format: True for (B, Seq, Feature), False for (Seq, B, Feature).
            norm_first (bool): Whether to apply normalization before attention/feedforward.
            bias (bool): Use bias in the Transformer layers.
            max_length (int): Maximum input sequence length (for positional embedding).
            linear_output_size (int): Output size of the final linear layer.
            linear_bias (bool): Use bias in the linear layer.

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
                "num_layers": ("INT", {"default": 6, "min": 1, "max": 256}),
                "vocabulary_size": ("INT", {"default": 10000, "min": 1, "max": 1e6}),
                "hidden_size": ("INT", {"default": 512, "min": 1, "max": 1e6}),
                "nhead": ("INT", {"default": 8, "min": 1, "max": 32}),
                "dim_feedforward": ("INT", {"default": 2048, "min": 1, "max": 1e6}),
                "dropout": ("FLOAT", {"default": 0.1, "min": 0, "max": 1}),
                "nonlinearity": (("gelu", "relu"),),
                "layer_norm_eps": ("FLOAT", {"default": 1e-5, "min": 1e-9, "max": 1e-1, "step": 1e-9}),
                "batch_first": ("BOOLEAN", {"default": True}),
                "norm_first": ("BOOLEAN", {"default": False}),
                "bias": ("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 1e6}),
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
          num_layers: int,
          vocabulary_size: int,
          hidden_size: int,
          nhead: int,
          dim_feedforward: int,
          dropout: float,
          nonlinearity: str,
          layer_norm_eps: float,
          batch_first: bool,
          norm_first: bool,
          bias: bool,
          max_length: int,
          linear_output_size: int,
          linear_bias: bool
        ) -> Tuple[nn.Module]:
        """
        Constructs a Transformer model with a linear output layer.

        Args:
            num_layers (int): Number of encoder layers.
            vocabulary_size (int): Vocabulary size of the tokens.
            hidden_size (int): Hidden size (d_model) of the Transformer.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Feedforward dimension.
            dropout (float): Dropout rate.
            nonlinearity (str): Activation function ('relu' or 'gelu').
            layer_norm_eps (float): Epsilon for layer normalization.
            batch_first (bool): Input shape format: True for (B, Seq, Feature), False for (Seq, B, Feature).
            norm_first (bool): Whether normalization comes before sublayers.
            bias (bool): Use bias in Transformer layers.
            max_length (int): Maximum sequence length.
            linear_output_size (int): Size of linear layer output.
            linear_bias (bool): Use bias in linear layer.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = EmbeddingTransformerLinear(
                num_encoder_layers=num_layers,
                vocabulary_size=vocabulary_size,
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=nonlinearity,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                bias=bias,
                max_length=max_length,
                linear_output_size=linear_output_size,
                linear_bias=linear_bias
            )
            return (model,)

