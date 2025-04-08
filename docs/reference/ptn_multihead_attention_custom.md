# Ptn Multihead Attention Custom
A Multihead attention model.

    Args:
        embed_dim (int): Hidden dimension (d_model) for each token of the Transformer.
        num_heads  (int): Number of attention heads.
        dropout (float): Dropout rate for output.
        bias (bool): If True, bias will be added to the linear projections of query, key, value, and the final output.
        kdim (int): Dimension of k.
        vdim (int): Dimension of v.
        batch_first (bool): Input shape format: True for (B, Seq, Feature), False for (Seq, B, Feature).

## Input
| Name | Data type |
|---|---|
| embed_dim | Int |
| num_heads | Int |
| dropout | Float |
| bias | Boolean |
| kdim | Int |
| vdim | Int |
| batch_first | Boolean |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
