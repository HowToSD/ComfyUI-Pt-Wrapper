# Sp Encode
SentencePiece wrapper for encoding text into token ID tensors.

    Args:  
        spmodel (SentencePieceProcessor): Trained SentencePiece model.  
        padding (bool): Whether to pad sequences to equal length.  
        padding_method (str): 'longest' or 'max_length'.  
            - 'longest': pad to the longest sequence length after truncation in this batch.  
            - 'max_length': pad all sequences to the fixed length `max_length`.  
        padding_value (Union[int, torch.Tensor]): A scalar value or scalar tensor used for padding.  
        truncation (bool): Whether to truncate sequences to `max_length`.  
        max_length (int): The target length for truncation or padding.  

    Returns:  
        PTCALLABLE: A tuple containing a callable that accepts a sentence or list of
        sentences and returns token ID tensor(s) and attention mask(s).

## Input
| Name | Data type |
|---|---|
| spmodel | Spmodel |
| padding | Boolean |
| padding_method |  |
| padding_value | Int |
| truncation | Boolean |
| max_length | Int |

## Output
| Data type |
|---|
| Ptcallable |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
