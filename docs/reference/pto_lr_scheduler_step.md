# Pto Lr Scheduler Step
Creates a StepLR learning rate scheduler for an optimizer.

Parameters  

**optimizer**  
  (torch.optim.Optimizer) – Wrapped optimizer whose learning rate needs scheduling.

**step_size**  
  (int) – Period of learning rate decay.

**gamma**  
  (float) – Multiplicative factor of learning rate decay.

## Input
| Name | Data type |
|---|---|
| optimizer | Ptoptimizer |
| step_size | Int |
| gamma | Float |

## Output
| Data type |
|---|
| Ptlrscheduler |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
