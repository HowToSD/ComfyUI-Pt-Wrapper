# Ptn Hf Lora Classification Model
A binary classification model containing a Hugging Face pretrained Transformer model.

    Args:  
        model_name (str): Name or path of the pretrained Hugging Face model.  
        use_mean_pooling (bool): If True, average token embeddings using the attention mask.  
                                Otherwise, use pooler output (if available) or the CLS token.  
        dropout (float): Dropout rate applied before the final classification layer.  
                        Set to 0.0 to disable dropout. Must be ≥ 0.  
        lora_r (int): LoRA rank parameter (dimension of the low-rank matrices).  
        lora_alpha (int): LoRA scaling factor. The adapted weight is added as:  
                        (lora_alpha / lora_r) * (A @ B), where  
                        A is a (m × r) matrix and B is a (r × n) matrix.  
        lora_dropout (float): Dropout applied to the input of the LoRA layers.

## Input
| Name | Data type |
|---|---|
| model_name | String |
| use_mean_pooling | Boolean |
| dropout | Float |
| lora_r | Int |
| lora_alpha | Int |
| lora_dropout | Float |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
