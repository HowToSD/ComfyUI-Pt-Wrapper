# Ptn RNN Linear
A recurrent neural network (RNN) model with a linear head.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of output features of the hidden layer matrix.
        num_layers (int): Number of hidden layers in RNN.
        nonlinearity (str): Activation function to apply after each hidden layer. Specify 'tanh' or 'relu'.
        bias (bool): Use bias or not.
        batch_first (bool): Indicates the shape of the input. Input is assumed to be a rank-3 tensor. If True, the shape is [Batch, Seq, Token]. If False, the shape is [Seq, Batch, Token]. Note that default is set to `True` unlike PyTorch's RNN default model parameter.
        dropout (float): Dropout probability to be applied to the output of intermediate layers if non-zero. This will not be applied to the last layer's output.
        bidirectional (bool): Processes the input in a backward direction and append the result to the output of the forward direction.
        linear_output_size (int): The number of output features of the linear layer.
        linar_bias (bool): Use bias or not in linear layer.

## Input
| Name | Data type |
|---|---|
| input_size | Int |
| hidden_size | Int |
| num_layers | Int |
| nonlinearity |  |
| bias | Boolean |
| batch_first | Boolean |
| dropout | Float |
| bidirectional | Boolean |

## Output
| Data type |
|---|
| Ptmodel |

<HR>
Category: PyTorch wrapper - Model

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
