# Ptn Conv
A convolutional model consisting of a single conv2d layer.  

        Args:
            in_channels (int): The number of input channels  
            out_channels (int): The number of output channels  
            kernel_size (str): The size of the kernel  
            stride (str): The stride to be used for sliding the kernel  
            padding (str): The amount of padding added to the input  
            dilation (str): The distance between adjacent elements in the kernel to adjust receptive fields.  
            groups (str): The number of groups used to divide input and output channels for separate processing.  
            bias (bool): If `True`, adds a learnable bias to the output  
        padding_mode (str): Specifies how padding values are set in the input before convolution.  
t. 
            bias (bool): Use bias or not.

## Input
| Name | Data type |
|---|---|
| in_channels | Int |
| out_channels | Int |
| kernel_size | String |
| stride | String |
| padding | String |
| dilation | String |
| groups | Int |
| bias | Boolean |
| padding_mode | Zeros |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
