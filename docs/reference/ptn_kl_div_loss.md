# Ptn KL Div Loss
A class to compute the KL divergence loss.

    Args:  
        reduction (str): Reduction method.  
                          `none`: no reduction will be applied done.   
                          `batchmean`: computes the sum then divides by the input size (recommended in the PyTorch document).
                          `mean`: computes the mean.  
                          `sum`: computes the sum.  

        log_target (bool): Set to `True` if the target is in log probability instead of probability.
                           The input (y_hat) must always be in log-probability form, regardless of this flag.

## Input
| Name | Data type |
|---|---|
| reduction | None |
| log_target | Boolean |

## Output
| Data type |
|---|
| Ptloss |

<HR>
Category: PyTorch wrapper - Loss function

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
