# Pto Lr Scheduler Reduce On Plateau
Creates a reduce-on-plateau learning rate scheduler for an optimizer.

Parameters  

**optimizer**  
  (torch.optim.Optimizer) – Wrapped optimizer whose learning rate needs scheduling.

**grace_period**  
  (int) – Number of epochs to monitor the val loss reduction. If loss does not decrease within grace period epochs, learning rate will be reduced by multiplying the gamma.  E.g. if gamma is set to 0.5, new learning rate will be 0.5 * current learning rate.
   
**gamma**  
  (float) – Learning rate multiplier to be applied when grace period is up.

## Input
| Name | Data type |
|---|---|
| optimizer | Ptoptimizer |
| grace_period | Int |
| gamma | Float |

## Output
| Data type |
|---|
| Ptlrscheduler |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
