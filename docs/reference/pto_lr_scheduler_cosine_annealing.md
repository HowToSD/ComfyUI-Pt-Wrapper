# Pto Lr Scheduler Cosine Annealing
Creates a cosine annealing learning rate scheduler for an optimizer.

https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
lists:

eta_t = eta_min + (1/2) * (eta_max - eta_min) * (1 + cos((t_cur / t_max) * pi))

eta_min is set to 0 by default, so the equation simplifies to:

eta_t = (1/2) * eta_max * (1 + cos((t_cur / t_max) * pi))

When t_cur = 0:
  (1 + cos((t_cur / t_max) * pi)) evaluates to (1 + cos(0 * pi)) = (1 + 1) = 2

When t_cur = t_max:
  (1 + cos((t_cur / t_max) * pi)) evaluates to (1 + cos(1 * pi)) = (1 - 1) = 0

Therefore, eta_t value changes as follows:
* Starts from (1/2) * eta_max * 2, which simplifies to eta_max
* Ends at 0

Parameters  

**optimizer**  
  (torch.optim.Optimizer) – Wrapped optimizer whose learning rate needs scheduling.

**num_epochs**  
  (int) – Number of epochs. This maps to the T_max parameter for CosineAnnealingLR internally.

**minimum_lr**  
  (float) – Minimum learning rate.

## Input
| Name | Data type |
|---|---|
| optimizer | Ptoptimizer |
| num_epochs | Int |
| minimum_lr | Float |

## Output
| Data type |
|---|
| Ptlrscheduler |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
