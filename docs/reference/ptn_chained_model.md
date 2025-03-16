# Ptn Chained Model
Constructs a chained PyTorch model.

### Fields:  
- model_a: First model in the chain.
- model_b: Second model in the chain.
- closure: Optional differentiable function to be called at the end of forward.

## Input
| Name | Data type |
|---|---|
| model_a | Ptmodel |
| model_b | Ptmodel |
| closure | Ptcallable |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
