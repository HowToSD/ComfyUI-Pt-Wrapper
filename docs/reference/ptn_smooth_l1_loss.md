# Ptn Smooth L1 Loss
A class to compute the Smooth L1 loss.

Please see [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html) for details and exact formulation.


    Args:  
        reduction (str): Reduction method.  
                          `none`: no reduction will be applied done.   
                          `mean`: computes the mean.  
                          `sum`: computes the sum.  
        beta (float): The threshold used to switch between adjusted L2 and L1 loss.

## Input
| Name | Data type |
|---|---|
| reduction | Mean |
| beta | Float |

## Output
| Data type |
|---|
| Ptloss |

<HR>
Category: PyTorch wrapper - Loss function

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
