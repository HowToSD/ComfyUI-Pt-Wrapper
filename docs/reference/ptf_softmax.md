# Ptf Softmax
A PyTorch wrapper for the softmax activation function.

This class provides a callable that applies the softmax activation function 
from `torch.nn.functional`.

Internally, this node calls torch.nn.functional.softmax(dim=-1).
See https://github.com/pytorch/pytorch/issues/1020 to find out more about the meaning of dim.

## Output
| Data type |
|---|
| Ptcallable |

<HR>
Category: PyTorch wrapper - Callable

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
