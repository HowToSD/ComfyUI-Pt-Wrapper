# Ptn Cross Entropy Loss
A class to compute the cross entropy loss.

Unlike Ptn NLL Loss node which takes log probabilities for input, input to this node needs to be logits.  You can consider this node as a combination of log softmax and NLL.

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
