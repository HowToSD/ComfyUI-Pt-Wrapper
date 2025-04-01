import os
import sys
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from .utils import set_requires_grad

class PtTrainFineTuneClassificationTransformerModel:
    """
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

    category: PyTorch wrapper - Training
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types required for training the model.

        Returns:
            Dict[str, Any]: A dictionary specifying required and optional input types.
        """
        return {
            "required": {
                "model": ("PTMODEL", {}),
                "train_loader": ("PTDATALOADER", {}),
                "optimizer": ("PTOPTIMIZER", {}),
                "loss_function": ("PTLOSS", {}),
                "epochs": ("INT", {"default":2, "min":1, "max":1e6}),
                "freeze_pretrained_module_epochs": ("INT", {"default":1, "min":1, "max":1e6}),
                "use_gpu": ("BOOLEAN", {"default":False}),
                "early_stopping": ("BOOLEAN", {"default":False}),
                "early_stopping_rounds": ("INT", {"default":10, "min":1, "max":1000}),
                "output_best_val_model": ("BOOLEAN", {"default":True}),
                "classification_metrics": ("BOOLEAN", {"default":True})
            },
            "optional": {
                "val_loader": ("PTDATALOADER", {})
            }
        }

    RETURN_NAMES: tuple = ("Model", "train loss", "val loss")
    RETURN_TYPES: tuple = ("PTMODEL","TENSOR","TENSOR")
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self, 
          model: nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          loss_function: nn.Module,
          epochs: int,
          freeze_pretrained_module_epochs,
          use_gpu: bool, 
          early_stopping: bool,
          early_stopping_rounds: int,
          output_best_val_model:bool,
          classification_metrics:bool,
          val_loader: Optional[torch.utils.data.DataLoader] = None,
          ) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        """
        Trains an Transformer model.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader containing the training dataset.
            optimizer (PTOPTIMIZER): Optimizer used for training.  Only Adam and AdamW are supported.
            loss (torch.nn.Module): The loss class.
            epochs (int): Number of training epochs.
            freeze_pretrained_module_epochs (int) : Number of epochs to freeze pretrained module.
            use_gpu (bool): Whether to use GPU for training.
            early_stopping (bool): Whether to turn on early stopping
            early_stopping_rounds (int): Specifies the number of rounds to monitor the evaluation loss when early_stopping is enabled and validation data is specified. If the loss does not decrease compared to the current best loss within this period, training will be stopped.
            output_best_val_model (bool): Whether to outputs the model with the best val loss or the model from the last epoch.
            classification_metrics (bool): If True and if val_loader is set, print out classification metrics.  This is only valid for classification model training.
            val_loader (Optional[torch.utils.data.DataLoader]): DataLoader for validation dataset (optional).

        Returns:
            Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]: A tuple containing the trained model, 
            training loss history, and validation loss history.
        """
        checkpoint_dir = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "tmp"))
        print(f"Checkpoint tmp directory is set to {checkpoint_dir}")
        if os.path.exists(checkpoint_dir) is False:
            os.makedirs(checkpoint_dir)
            print(f"Created {checkpoint_dir}")
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

        best_val_loss = None
        rounds_since_best_val_loss = 0

        with torch.inference_mode(False):
            train_loss = []
            val_loss = []
            if use_gpu:
                model.to("cuda")

            # Train the model
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                total_samples = len(train_loader.dataset)

                if epoch < freeze_pretrained_module_epochs:
                    set_requires_grad(model.llm_model, False)  # freeze encoder
                elif epoch == freeze_pretrained_module_epochs:
                    set_requires_grad(model.llm_model, True)   # unfreeze encoder

                    # Rebuild optimizer with separate LR for pretrained encoder and classifier head
                    if isinstance(optimizer, torch.optim.Adam):
                        lr = optimizer.defaults.get("lr", 1e-3)
                        betas = optimizer.defaults.get("betas", (0.9, 0.999))
                        eps = optimizer.defaults.get("eps", 1e-8)
                        weight_decay = optimizer.defaults.get("weight_decay", 0)

                        optimizer = torch.optim.Adam([
                            {"params": model.llm_model.parameters(), "lr": lr * 0.2},
                            {"params": model.linear.parameters(), "lr": lr}
                        ], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

                    elif isinstance(optimizer, torch.optim.AdamW):
                        lr = optimizer.defaults.get("lr", 1e-3)
                        betas = optimizer.defaults.get("betas", (0.9, 0.999))
                        eps = optimizer.defaults.get("eps", 1e-8)
                        weight_decay = optimizer.defaults.get("weight_decay", 0)

                        optimizer = torch.optim.AdamW([
                            {"params": model.llm_model.parameters(), "lr": lr * 0.2},
                            {"params": model.linear.parameters(), "lr": lr}
                        ], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

                    else:
                        raise ValueError("Only Adam and AdamW are supported for optimizer re-instantiation.")

                for x, y in train_loader:
                    if use_gpu:
                        if isinstance(x, list):
                            x = [e.to("cuda") for e in x]
                        else:
                            x = x.to("cuda")
                        y = y.to("cuda")
                    optimizer.zero_grad()

                    y_hat = model(x)
                    if y.dim() == 1:
                        y = torch.unsqueeze(y, dim=-1)
                        y = y.to(torch.float32)

                    loss = loss_function(y_hat, y)
                    loss.backward()
                    optimizer.step()
 
                    batch_size = y.size(0)
                    batch_loss = loss.item() * batch_size
                    total_loss += batch_loss

                epoch_loss = total_loss / total_samples
                print(f"Epoch (train) {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
                train_loss.append(epoch_loss)
                
                if val_loader:
                    with torch.no_grad():
                        all_preds = []
                        all_labels = []

                        model.train(False)  # TODO: Change to model.*e*v*a*l() once Comfy's security checker is fixed.
                        total_loss = 0
                        total_samples = len(val_loader.dataset)
                        for x, y in val_loader:
                            if use_gpu:
                                if isinstance(x, list):
                                    x = [e.to("cuda") for e in x]
                                else:
                                    x = x.to("cuda")
                                y = y.to("cuda")

                            y_hat = model(x)
                            if y.dim() == 1:
                                y = torch.unsqueeze(y, dim=-1)
                                y = y.to(torch.float32)

                            loss = loss_function(y_hat, y)
                            batch_size = y.size(0)
                            batch_loss = loss.item() * batch_size
                            total_loss += batch_loss

                            if classification_metrics and y.squeeze().dim() == 1 :
                                # Accuracy collection for binary classification
                                preds = (F.sigmoid(y_hat) > 0.5).int().detach().cpu().numpy()
                                labels = y.int().detach().cpu().numpy()
                                all_preds.extend(preds)
                                all_labels.extend(labels)

                        epoch_loss = total_loss / total_samples
                        print(f"Epoch (val) {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
                        val_loss.append(epoch_loss)

                        if classification_metrics:
                            if len(all_labels):
                                # Compute accuracy
                                acc = accuracy_score(all_labels, all_preds)
                                print(f"Validation Accuracy: {acc:.4f}")
                            else:
                                print("Accuracy was not computed as the label is not scalar.")

                        if best_val_loss is None or epoch_loss < best_val_loss:
                            best_val_loss = epoch_loss
                            rounds_since_best_val_loss = 0
                            torch.save(model.state_dict(), best_checkpoint_path)
                            print(f"Updated best val loss. Saved the model as {best_checkpoint_path}")
                        else:
                            rounds_since_best_val_loss += 1
                            if early_stopping:
                                if rounds_since_best_val_loss >= early_stopping_rounds:
                                    print(f"Val loss did not improve in {early_stopping_rounds} rounds. Stopping training.")
                                    break

            if use_gpu:
                model.to("cpu")

            if val_loader:
                if epoch_loss > best_val_loss and output_best_val_model:
                    sd = torch.load(best_checkpoint_path, weights_only=True)
                    model.load_state_dict(sd)
                    print("Loaded the best model for output")
                else:
                    print("Using the final epoch model for output.")

            return (model, 
                    torch.tensor(train_loss, dtype=torch.float32), 
                    torch.tensor(val_loss, dtype=torch.float32))