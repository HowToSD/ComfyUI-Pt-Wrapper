# Ptn Hf Fine Tuned Classification Model
A binary classification model containing a Hugging Face pretrained Transformer model.

        Args:
        model_name (str): Name or path of the pretrained Hugging Face model.
        use_mean_pooling (bool): If True, average token embeddings using the attention mask.
        Otherwise, use pooler output or CLS token embedding.
        dropout (float): Dropout rate applied before the final linear layer.
                         Set to 0.0 to disable dropout. Must be ≥ 0.

## Input
| Name | Data type |
|---|---|
| model_name | String |
| use_mean_pooling | Boolean |
| dropout | Float |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
