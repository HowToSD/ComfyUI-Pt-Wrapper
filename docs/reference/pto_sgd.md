# Pto SGD
Instantiates the SGD optimizer.

Parameters  
        model (PTMODEL): The PyTorch model whose parameters will be optimized.  
        learning_rate (float): The learning rate for the AdamW optimizer.  
        momentum (float):  Coefficient to apply to the past gradient to adjust the contribution of past gradients  
        dampening (float): Coefficient to adjust the contribution of current gradient.  
        weight_decay (float): The weight decay parameter.  
        nesterov (bool): If True, uses the Nesterov version.

## Input
| Name | Data type |
|---|---|
| model | Ptmodel |
| learning_rate | Float |
| momentum | Float |
| dampening | Float |
| weight_decay | Float |
| nesterov | Boolean |

## Output
| Data type |
|---|
| Ptoptimizer |

<HR>
Category: PyTorch wrapper - Optimizer

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
