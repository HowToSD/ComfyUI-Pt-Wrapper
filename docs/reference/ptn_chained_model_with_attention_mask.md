# Ptn Chained Model With Attention Mask
A chained model that sequentially applies `model_a` and `model_b`,
optionally followed by a closure function.

### Fields:  
- `model_a`: First model in the chain.
- `model_b`: Second model in the chain.
- `model_a_mask_req`: True if model_a requires attention_mask for input
- `model_b_mask_req`: True if model_b requires attention_mask for input
- closure: Optional differentiable function to be called at the end of forward.

## Input
| Name | Data type |
|---|---|
| model_a | Ptmodel |
| model_b | Ptmodel |
| model_a_mask_req | Boolean |
| model_b_mask_req | Boolean |
| closure | Ptcallable |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
