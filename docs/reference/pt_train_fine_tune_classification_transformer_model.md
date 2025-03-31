# Pt Train Fine Tune Classification Transformer Model
Fine-tunes a classification Transformer model using a given dataset, loss function, optimizer, and number of epochs with learning rate decay.  

   Args:  
        model (PTMODEL): The PyTorch model to be trained.  
        train_loader (torch.utils.data.DataLoader): DataLoader containing the training dataset.  
        optimizer (PTOPTIMIZER): Optimizer used for training.  Only Adam and AdamW are supported.
        loss (PTLOSS): The loss class.  
        epochs (int): Number of training epochs.  
        freeze_pretrained_module_epochs (int) : Number of epochs to freeze pretrained module.
        use_gpu (bool): Whether to use GPU for training.  
        early_stopping (bool): Whether to turn on early stopping  
        early_stopping_rounds (int): Specifies the number of rounds to monitor the evaluation loss when early_stopping is enabled and validation data is specified. If the loss does not decrease compared to the current best loss within this period, training will be stopped.  
        output_best_val_model (bool): Whether to outputs the model with the best val loss or the model from the last epoch.  
        classification_metrics (bool): If True and if val_loader is set, print out classification metrics.  This is only valid for classification model training.  
        val_loader (PTDATALOADER): DataLoader for validation dataset (optional).

## Input
| Name | Data type |
|---|---|
| model | Ptmodel |
| train_loader | Ptdataloader |
| optimizer | Ptoptimizer |
| loss_function | Ptloss |
| epochs | Int |
| freeze_pretrained_module_epochs | Int |
| use_gpu | Boolean |
| early_stopping | Boolean |
| early_stopping_rounds | Int |
| output_best_val_model | Boolean |
| classification_metrics | Boolean |
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
