# Ptn Instance Norm 2d
A normalization model to normalize elements over spatial axes within each channel for each sample.

    Args:
        num_features (int): Specifies the size of channel axis.
            - For an image input of shape `[8, 4, 256, 256]`, specify 4.

        affine (bool): If `True`, applies a learnable scaling factor and bias to the normalized elements.

        track_running_stats (bool): If `True`, use exponential moving average (EMA) to store mean and std to be used in eval for normalization. For training, current sample data is always used for mean and std irrespective of this flag.
         
        momentum (float): The EMA coefficient used if track_running_stats is set to True. Higher value gives more weight to the current statistics (mean & std).

## Input
| Name | Data type |
|---|---|
| num_features | Int |
| affine | Boolean |
| track_running_stats | Boolean |
| momentum | Float |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
