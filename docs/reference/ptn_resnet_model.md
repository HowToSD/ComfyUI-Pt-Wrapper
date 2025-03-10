# Ptn Resnet Model
A Resnet model consisting of multiple Resnet layers.  

### Fields:  
- `input_dim`: A string representing the input dimensions in the format "(C,H,W)".  
  Example: `"(3,32,32)"` for a 3-channel 28x28 image.  

- `output_dim`: An integer specifying the number of output features, which should match the number of target classes in classification.
  Default: `10`, Min: `1`, Max: `1e6`.  

- `num_blocks`: Number of Resnet blocks for 64 channel blocks. Same number of blocks will be created for 128 channel and 256 channel blocks.

## Input
| Name | Data type |
|---|---|
| input_dim | String |
| output_dim | Int |
| num_blocks | Int |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Data Analysis Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
