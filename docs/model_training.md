# Model Training

You can train a PyTorch model without any coding.
Training is as simple as a single click!  
Currently, following archituctures are supported (node names are in parentheses):
* MLP (Ptn Linear Model)
* Convolutional Network (Ptn Conv Model)
* ResNet (Ptn Resnet Model)

You can adjust the model configuration including output dimension on the UI.

The system is designed to be modular. For example, to switch from a linear model to a convolutional model, you only need to replace the model node (see screenshot below).

You can also build a model from scratch by sequentially combining basic models such as `Conv2D` and `Linear` by using the **Pt Chained Model** node.  See [Building a Model from Scratch](building_a_model_from_scratch.md) for details.

You can choose to:
* Use your own images
* Use a public dataset

## Using Your Own Images

Check out [the dog and cat classification tutorial](dog_cat_classification_model_training.md).

## Public Datasets

Currently, only Fashion MNIST and CIFAR-10 have been tested. Other public datasets have not been verified yet, though they may work.  
If you try one and it doesn't work, please create an issue so I can investigate adding support.

To use these datasets, simply drag and drop the workflow files below into ComfyUI. The model will be downloaded automatically.  Make sure that you update the directory to save the trained model.

Linear model  
* Training: examples/workflows/fashion_mnist_train.json  
* Evaluation: examples/workflows/fashion_mnist_eval.json

Convolutional model  
* Training: examples/workflows/fashion_mnist_train_conv.json  
* Evaluation: examples/workflows/fashion_mnist_eval_conv.json

ResNet model
**Training**
* examples/workflows/cifar10_train_v8.json
* examples/workflows/cifar10_train_v11_2nd_run.json
  
**Evaluation**
* examples/workflows/cifar10_eval_v8.json
* examples/workflows/cifar10_eval_v11_2nd_run.json

This train_v8 model achieved 93.12% accuracy & train_v11_2nd_run model achieved 94.34 on CIFAR-10 (first run was 94.27%). train_v11_2nd_run added cosine annealing learning rate scheduler while train_v8 used fixed learning rate. While not state-of-the-art, these are strong results. Due to random initialization, data augmentation, and dataset shuffling, accuracy may vary slightly between runs. If your accuracy is significantly lower, consider running it multiple times or checking for potential issues in the training setup.

**Note: Before you try these workflows, please install ComfyUI-Data-Analysis extension.**

### Training workflow
**Linear model**
![Train](images/fashion_mnist_train.png)

**Conv model**
![Train](images/conv_train.png)

*Close up of the conv node*
![Train](images/conv_train2.png)

**ResNet model**
![Train](images/resnet_train.png)

### Eval workflow
**Linear model**
![Eval](images/fashion_mnist_eval.png)

**Conv model**
![Eval](images/conv_eval.png)

**ResNet model**
![Eval](images/resnet_eval.png)

# Using the Single-Layer Linear Model

A single-layer linear model is included to illustrate how to connect various nodes.  
A workflow containing this model and related nodes is in the following file:  
`examples/workflows/simple_linear_regression_train.json`.

This workflow illustrates:
- How to generate linear regression test data
- How to plot the data
- How to train a model by connecting it to an optimizer
- How to make predictions using a trained model

If you run training, you will notice that the loss remains in the 20s–30s instead of approaching 0, as you might expect.

Do you know why this happens?

# Using MLP for Small Dataset Classification

MLP can be used for small dataset classification.  
Below is an example workflow for classifying the classic Iris dataset:  
`examples/workflows/iris_classification_train_eval.json`.

According to [the UCI repository](https://archive.ics.uci.edu/dataset/53/iris), the neural network baseline accuracy for this dataset ranges from 92.105% to 100%. Using this workflow, you can expect around 96% accuracy, occasionally reaching 100%, which aligns with the baseline.

# Specifying a loss function for training
If you want to specify a loss function for training instead of using the default one in the trainer, use the `Pt Train Model` node. Create a loss function node (e.g., `Pt Cross Entropy Loss`) and connect it to the trainer node.
Refer to `examples/workflows/train_model_with_custom_loss_example.json` for an example. (This workflow is intended to demonstrate node connections and is not optimized for high CIFAR10 accuracy. It runs for only 10 epochs to allow a quick check and reaches 88.9% accuracy.) You should adjust the hyperparameters according to your specific goals.

# Training RNN
Refer to [Training RNN for text classification](training_rnn_for_classification.md)

# Training Transformer
Refer to [Training Transformer for text classification](training_transformer_for_classification.md)