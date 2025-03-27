# Pt Tokenizer
The tokenizer to encode a string or a list of string to token IDs.

    Args:  
        encode (PTCALLABLE): The reference to a token encoder function.  
        text_list (PYLIST): The python list of strings to tokenize.  

    Returns:  
        A tuple containing the token IDs and the mask in torch.Tensor.

## Input
| Name | Data type |
|---|---|
| encode | Ptcallable |
| text_list | Pylist |

## Output
| Data type |
|---|
| Tensor |
| Tensor |

<HR>
Category: PyTorch wrapper - Callable

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
