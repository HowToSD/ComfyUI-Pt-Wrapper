# Ptn Embedding Transformer Linear
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

## Input
| Name | Data type |
|---|---|
| num_layers | Int |
| vocabulary_size | Int |
| hidden_size | Int |
| nhead | Int |
| dim_feedforward | Int |
| dropout | Float |
| nonlinearity |  |
| layer_norm_eps | Float |
| batch_first | Boolean |
| norm_first | Boolean |
| bias | Boolean |
| max_length | Int |
| linear_output_size | Int |
| linear_bias | Boolean |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
