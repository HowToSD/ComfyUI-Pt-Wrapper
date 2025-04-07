# Ptn Residual Connection Model
A model that saves the input and add to the output of the specified model.
optionally followed by a closure function.

### Fields:  
- `model`: A model to process input.
- `closure`: Optional differentiable function to be called at the end of forward.

## Input
| Name | Data type |
|---|---|
| model | Ptmodel |
| closure | Ptcallable |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
