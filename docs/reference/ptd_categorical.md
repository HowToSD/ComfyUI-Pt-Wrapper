# Ptd Categorical
Instantiates a Categorical distribution object from the input probabilities or logits. You have to specify one of them and not both.

Args:
    probs (str): Probabilities of the distsribution. You can specify a scalar or a tuple of float.  
    logits (str): Logits of the distribution. You can specify a scalar or a tuple of float.

## Input
| Name | Data type |
|---|---|
| probs | String |
| logits | String |

## Output
| Data type |
|---|
| Ptdistribution |

<HR>
Category: PyTorch wrapper - Distribution

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
