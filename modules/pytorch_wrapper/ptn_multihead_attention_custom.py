from typing import Any, Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
from einops import einops

class PtnMultiheadAttentionCustomDef(nn.Module):
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
        kdim: int,
        vdim: int,
        batch_first: bool
    ) -> None:
        super().__init__()

        # Compute dimensions
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Projection matrix
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.batch_first = batch_first


    def forward(self,
                inputs: torch.Tensor,
                mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim) if batch_first is True.
            mask:Optional[torch.Tensor]: An optional attention mask where 1 indicates a valid token and 0 indicates padding.
                This is the inverse of PyTorch's expected key_padding_mask, which masks out positions where the value is True.

        Returns:
            torch.Tensor: Output tensor after multi-head self-attention.
        """
        if inputs.dim() != 3:
            raise ValueError("Inputs is not a rank 3 tensor.")

        x = inputs

        if not self.batch_first:  # (seq, batch, features) to (batch, seq, feature)
            x = einops.rearrange(x, 's b f -> b s f')

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split feature axis into heads × (feature per head).
        # The size of this per-head feature is denoted by "d" (aka "head_dim").
        # Rearrange so that dimensions are (batch, heads, seq, d).
        # (h d) means h * d.
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        # dot product q @ k.T / sqrt(head_dim)
        kt = torch.transpose(k, -1, -2)
        dot = torch.matmul(q, kt) / math.sqrt(self.head_dim)  # torch.sqrt requires a tensor

        # q @ k = (b h s d) @ (b h d s) = (b h s s)
        if dot.size() != (q.size(0), q.size(1), q.size(2), q.size(2)):
            raise ValueError(f"Unexpected size for q@k.T: {dot.size()}")

        # Apply mask (1 = valid token, 0 = padding)
        # If token contains 0, the attention score for the cell
        # will be replaced with -inf, which will be converted to 0
        # in softmax.
        # e.g. one row of attention score
        # pos  0  1  2 ... 1023
        #      5  2  -5     -20
        # mask
        #      1  1  0        0
        # result
        #      5  2   -inf   -inf
        if mask is not None:
            mask = mask.to(torch.bool)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            dot = dot.masked_fill(mask == 0, float('-inf'))

        # Normalize
        dot = torch.softmax(dot, -1)

        # Apply attention weights to value vectors
        v = torch.matmul(dot, v)

        # Rearrange
        v = einops.rearrange(v, 'b h s d -> b s (h d)')

        x = self.out_proj(v)

        if self.dropout:
            x = self.dropout(x)

        if not self.batch_first:
            x = einops.rearrange(x, 'b s f -> s b f')

        return x


class PtnMultiheadAttentionCustom:
    """
    A Multihead attention model.

        Args:
            embed_dim (int): Hidden dimension (d_model) for each token of the Transformer.
            num_heads  (int): Number of attention heads.
            dropout (float): Dropout rate for output.
            bias (bool): If True, bias will be added to the linear projections of query, key, value, and the final output.
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
            kdim (int): Dimension of k.
            vdim (int): Dimension of v.
            batch_first (bool): Input shape format: True for (B, Seq, Feature), False for (Seq, B, Feature).

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = PtnMultiheadAttentionCustomDef(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                kdim=kdim,
                vdim=vdim,
                batch_first=batch_first
            )
            return (model,)
