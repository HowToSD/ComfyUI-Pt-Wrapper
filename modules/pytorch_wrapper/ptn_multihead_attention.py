from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from typing import Tuple

class PtnMultiheadAttentionDef(nn.Module):
    """
    Implements MultiheadAttention wrapper.

    Attributes:
        embed_dim (int): Hidden dimension (d_model) for each token of the Transformer.
        num_heads  (int): Number of attention heads.
        dropout (float): Dropout rate for output.
        bias (bool): If True, bias will be added to the linear projections of query, key, value, and the final output.
        add_bias_kv (bool): If True, bias will be added to k and v sequences.
        add_zero_attn (bool): If True, a zero batch will be added  to k and v sequences.
        kdim (int): Dimension of k.
        vdim (int): Dimension of v.
        batch_first (bool): Input shape format: True for (B, Seq, Feature), False for (Seq, B, Feature).

    pragma: skip_doc
    """
    def __init__(self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: int,
        vdim: int,
        batch_first: bool
    ) -> None:
        super().__init__()

        self.model = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim) if batch_first is True.
            mask (torch.Tensor): Attention mask where 1 indicates a valid token and 0 indicates padding.
                This is the inverse of PyTorch's expected key_padding_mask, which masks out positions where the value is True.

        Returns:
            torch.Tensor: Output tensor after multi-head self-attention.
        """
        return self.model(
            inputs, inputs, inputs,
            need_weights=False,
            key_padding_mask=~mask.to(torch.bool)
        )[0]


class PtnMultiheadAttention:
    """
    A Multihead attention model.

        Args:
            embed_dim (int): Hidden dimension (d_model) for each token of the Transformer.
            num_heads  (int): Number of attention heads.
            dropout (float): Dropout rate for output.
            bias (bool): If True, bias will be added to the linear projections of query, key, value, and the final output.
            add_bias_kv (bool): If True, bias will be added to k and v sequences.
            add_zero_attn (bool): If True, a zero batch will be added  to k and v sequences.
            kdim (int): Dimension of k.
            vdim (int): Dimension of v.
            batch_first (bool): Input shape format: True for (B, Seq, Feature), False for (Seq, B, Feature).
            
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
                "embed_dim": ("INT", {"default": 512, "min": 1, "max": 1e6}),
                "num_heads": ("INT", {"default": 8, "min": 1, "max": 256}),
                "dropout": ("FLOAT", {"default": 0, "min": 0, "max": 1}),
                "bias": ("BOOLEAN", {"default": True}),
                "add_bias_kv": ("BOOLEAN", {"default": False}),
                "add_zero_attn": ("BOOLEAN", {"default": False}),
                "kdim": ("INT", {"default": 512, "min": 1, "max": 1e6}),
                "vdim": ("INT", {"default": 512, "min": 1, "max": 1e6}),
                "batch_first": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: int,
        vdim: int,
        batch_first: bool
        ) -> Tuple[nn.Module]:
        """
        Constructs a Multihead attention model.

        Args:
            embed_dim (int): Hidden dimension (d_model) for each token of the Transformer.
            num_heads  (int): Number of attention heads.
            dropout (float): Dropout rate for output.
            bias (bool): If True, bias will be added to the linear projections of query, key, value, and the final output.
            add_bias_kv (bool): If True, a learnable bias vector is prepended to the key and value sequences along the sequence dimension. Each bias vector has the same dimension as other sequence positions. Set this to False if you are unsure.
            add_zero_attn (bool): If True, a zero vector will be added to k and v sequences.
            kdim (int): Dimension of k.
            vdim (int): Dimension of v.
            batch_first (bool): Input shape format: True for (B, Seq, Feature), False for (Seq, B, Feature).

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = PtnMultiheadAttentionDef(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                kdim=kdim,
                vdim=vdim,
                batch_first=batch_first
            )
            return (model,)
