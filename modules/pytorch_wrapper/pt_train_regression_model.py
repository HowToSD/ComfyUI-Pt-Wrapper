import os
import sys
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PtTrainRegressionModel:
    """
    Trains a regression model using a given dataset, optimizer, and number of epochs.  

    early_stopping_rounds: Specifies the number of rounds to monitor the evaluation loss when early_stopping is enabled and validation data is specified. If the loss does not decrease compared to the current best loss within this period, training will be stopped.
    output_best_val_model: Whether to outputs the model with the best val loss or the model from the last epoch.

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
                "epochs": ("INT", {"default":1, "min":1, "max":1e6}),
                "use_gpu": ("BOOLEAN", {"default":False}),
                "early_stopping": ("BOOLEAN", {"default":False}),
                "early_stopping_rounds": ("INT", {"default":10, "min":1, "max":1000}),
                "output_best_val_model": ("BOOLEAN", {"default":True})
            },
            "optional": {
                "val_loader": ("PTDATALOADER", {}),
            }
        }

    RETURN_NAMES: tuple = ("Model", "train loss", "val loss")
    RETURN_TYPES: tuple = ("PTMODEL","TENSOR","TENSOR")
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self, model: nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          epochs: int, 
          use_gpu: bool, 
          early_stopping: bool,
          early_stopping_rounds: int,
          output_best_val_model:bool,
          val_loader: Optional[torch.utils.data.DataLoader] = None) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        """
        Trains a regression model.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader containing the training dataset.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            epochs (int): Number of training epochs.
            use_gpu (bool): Whether to use GPU for training.
            early_stopping (bool): Whether to turn on early stopping
            early_stopping_rounds (int): Specifies the number of rounds to monitor the evaluation loss when early_stopping is enabled and validation data is specified. If the loss does not decrease compared to the current best loss within this period, training will be stopped.
            output_best_val_model (bool): Whether to outputs the model with the best val loss or the model from the last epoch.
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

            loss_function = nn.MSELoss()  # apply mean loss

            # Train the model
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                total_samples = len(train_loader.dataset)
                for images, labels in train_loader:
                    if use_gpu:
                        images = images.to("cuda")
                        labels = labels.to("cuda")
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
 
                    batch_size = labels.size(0)
                    batch_loss = loss.item() * batch_size
                    total_loss += batch_loss

                epoch_loss = total_loss / total_samples
                
                print(f"Epoch (train) {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
                train_loss.append(epoch_loss)
                
                if val_loader:
                    with torch.no_grad():
                        model.train(False)  # TODO: Change to model.*e*v*a*l() once Comfy's security checker is fixed.
                        total_loss = 0
                        total_samples = len(val_loader.dataset)
                        for images, labels in val_loader:
                            if use_gpu:
                                images = images.to("cuda")
                                labels = labels.to("cuda")
                            outputs = model(images)
                            loss = loss_function(outputs, labels)
                            batch_size = labels.size(0)
                            batch_loss = loss.item() * batch_size
                            total_loss += batch_loss

                        epoch_loss = total_loss / total_samples
                        print(f"Epoch (val) {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
                        val_loss.append(epoch_loss)

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