# Ptdm Pmf Tensor
Computes the probability for the input distribution.  This nodes accepts a tensor so it can be used to compute pmf for multiple values contained in a tensor.

Args:
        distribution (torch.distributions.distribution.Distribution): Distribution.
        tens (torch.Tensor): k(s) in Tensor.

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
