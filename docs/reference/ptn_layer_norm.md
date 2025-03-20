# Ptn Layer Norm
A normalization model to normalize elements over specified axes.

    Args:
        normalized_shape (list): Specifies the shape of elements to normalize.  
            - For an input of shape `[8, 1024, 768]` (where axes represent `[batch_size, sequence_length, hidden_dim]`):
            - `normalize_shape=[768]` normalizes within each token (across the hidden dimension).
            - `normalize_shape=[1024, 768]` normalizes within each sample (across both sequence and token dimensions).  
            - For an image input of shape `[8, 4, 256, 256]` (where axes represent `[batch_size, channel, height, width]`):
            - `normalize_shape=[256, 256]` normalizes within each channel (across spatial dimensions).
            - `normalize_shape=[4, 256, 256]` normalizes within each sample (across channels and spatial dimensions).

        elementwise_affine (bool): If `True`, applies a learnable scaling factor to the normalized elements.

        bias (bool): If `True` and elementwise_affine is also `True`, applies a learnable bias to the normalized elements.

## Input
| Name | Data type |
|---|---|
| normalized_shape | String |
| elementwise_affine | Boolean |
| bias | Boolean |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
