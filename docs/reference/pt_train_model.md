# Pt Train Model
Trains a model using a given dataset, loss function, optimizer, and number of epochs with learning rate decay.  

early_stopping_rounds: Specifies the number of rounds to monitor the evaluation loss when early_stopping is enabled and validation data is specified. If the loss does not decrease compared to the current best loss within this period, training will be stopped.
output_best_val_model: Whether to outputs the model with the best val loss or the model from the last epoch.

Functionality-wise, this node is identical with PtTrainClassificationModelLr except that you can specify a custom loss function in this model where PtTrainClassificationModelLr uses nn.CrossEntropyLoss().

## Input
| Name | Data type |
|---|---|
| model | Ptmodel |
| train_loader | Ptdataloader |
| optimizer | Ptoptimizer |
| loss_function | Ptloss |
| epochs | Int |
| use_gpu | Boolean |
| early_stopping | Boolean |
| early_stopping_rounds | Int |
| output_best_val_model | Boolean |
| scheduler | Ptlrscheduler |
| val_loader | Ptdataloader |

## Output
| Data type |
|---|
| Ptmodel |
| Tensor |
| Tensor |

<HR>
Category: PyTorch wrapper - Training

ComfyUI Pt Wrapper Node Reference. © 2025 Hide Inada (HowToSD.com). All rights reserved.
