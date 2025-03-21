# Ptn NLL Loss
A model to compute the negative log likelihood (NLL) loss.

Unlike Ptn Cross Entropy which takes input in logits, input to this nodes needs to be in log-probabilities.

    Args:  
        reduction: (str): Reduction method.  
                          `none`: no reduction will be applied done.   
                          `mean`: computes the mean.  
                          `sum`: computes the sum.

## Input
| Name | Data type |
|---|---|
| reduction | Mean |

## Output
| Data type |
|---|
| Ptloss |

<HR>
Category: PyTorch wrapper - Loss function

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
