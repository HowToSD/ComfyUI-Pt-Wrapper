# Hf Tokenizer Encode
Hugging Face tokenizer wrapper for encoding text into token ID tensors.

    Args:  
        model_name (str): Name of the model.
        padding (bool): Whether to pad sequences to equal length.
        padding_method (str): 'longest' or 'max_length'.
            - 'longest': pad to the longest sequence length after truncation in this batch.
            - 'max_length': pad all sequences to the fixed length `max_length`.
        truncation (bool): Whether to truncate sequences to `max_length`.
        max_length (int): The target length for truncation or padding.  
                          If you specify 0, it uses model's maximum input length.

    Returns:  
        PTCALLABLE: A tuple containing a callable that accepts a sentence or list of
        sentences and returns token ID tensor(s) and attention mask(s).

## Input
| Name | Data type |
|---|---|
| model_name | String |
| padding | Boolean |
| padding_method |  |
| truncation | Boolean |
| max_length | Int |

## Output
| Data type |
|---|
| Ptcallable |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
