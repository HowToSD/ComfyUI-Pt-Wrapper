from typing import Any, Dict, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader


class PtDataLoaderFromTensors:
    """
    Creates a Torchvision Dataloader from a pair of tensors.
    
    Parameters  
    
    **x**  
      Specify a tensor name that contains x.
      
    **y**  
      Specify a tensor name that contains y.

    category: PyTorch wrapper - Training
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the required input types.  
        
        Returns:  
            Dict[str, Any]: A dictionary of required input types.  
        """
        return {
            "required": {
                "x": ("TENSOR", {}),
                "y": ("TENSOR", {}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1e6}),
                "shuffle": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES: tuple = ("PTDATALOADER",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self, 
          x: torch.Tensor,
          y: torch.Tensor,
          batch_size: int,
          shuffle: bool) -> Tuple:
        """
        Creates a Torchvision Dataloader from a pair of tensors.
        
        Args:  
            x (torch.Tensor): A tensor containing x.
            y (torch.Tensor): A tensor containing y.
            batch_size (int): The number of samples per batch.
            shuffle (bool): If True, shuffles the dataset before each epoch.

        Returns:  
            Tuple: A tuple containing the dataset instance.  
        """
        
        with torch.inference_mode(False):
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            return (dataloader,)
