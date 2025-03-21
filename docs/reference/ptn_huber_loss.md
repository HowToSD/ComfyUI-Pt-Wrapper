# Ptn Huber Loss
A class to compute the Huber loss.

Please see [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html) for details.


    Args:  
        reduction (str): Reduction method.  
                          `none`: no reduction will be applied done.   
                          `mean`: computes the mean.  
                          `sum`: computes the sum.  
        delta (float): The threshold used to switch between adjusted MSE and L1 loss.

## Input
| Name | Data type |
|---|---|
| reduction | Mean |
| delta | Float |

## Output
| Data type |
|---|
| Ptloss |

<HR>
Category: PyTorch wrapper - Loss function

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
