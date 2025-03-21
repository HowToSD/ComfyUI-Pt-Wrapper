# Ptn BCE Loss
A class to compute the binary cross entropy loss.

Unlike the Ptn BCE Loss With Logits node, which takes logits as input, this node expects probabilities. If the input consists of logits, apply a Sigmoid function first or use the Ptn BCE Loss With Logits node instead.

    Args:  
        reduction (str): Reduction method.  
                          `none`: no reduction will be applied done.   
                          `mean`: computes the mean.  
                          `sum`: computes the sum.

## Input
| Name | Data type |
|---|---|
| reduction |  |

## Output
| Data type |
|---|
| Ptloss |

<HR>
Category: PyTorch wrapper - Loss function

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
