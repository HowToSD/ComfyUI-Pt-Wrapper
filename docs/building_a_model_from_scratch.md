# Building a Model from Scratch

ComfyUI Pt Wrapper provides composite models that can be easily customized by adjusting parameters. However, you can also build a model from scratch by sequentially combining basic models such as `Conv2D` and `Linear`.

To chain two models, use the **Pt Chained Model** node. This node also supports specifying post-processing after the forward pass of the second model (e.g., activation functions). To define the post-processing step, connect a **Ptf** node (e.g., `Ptf ReLU`).

In some cases, preprocessing may be required before feeding data into the model. For instance, you can use the **Pt Pre Add Channel Axis** node to add a channel dimension to the input tensor or the **Pt Pre Flatten** node to flatten convolutional output before passing it to a linear layer.

The following example workflows demonstrate this approach:

- `./examples/workflows/fashion_mnist_eval_conv_chained_model.json`: Constructs a multi-layer convolutional network from scratch.
- `./examples/workflows/fashion_mnist_train_chained_model.json`: Uses the trained model for evaluation.
