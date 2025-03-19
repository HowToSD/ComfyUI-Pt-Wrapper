# Ptdm Icdf Tensor
Computes the inverse of cumulative distribution function for the input distribution. This nodes accepts a tensor so it can be used to compute cdf for multiple values contained in a tensor.

**Note**  
Icdf is not supported for all distributions in PyTorch.

Args:
        distribution (torch.distributions.distribution.Distribution): Distribution.
        tens (torch.Tensor): q in Tensor.

## Input
| Name | Data type |
|---|---|
| distribution | Ptdistribution |
| tens | Tensor |

## Output
| Data type |
|---|
| Tensor |

<HR>
Category: PyTorch wrapper - Distribution

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
