from typing import Any, Dict, Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset
import torch
from typing import Tuple

class SequentialTensorDataset(Dataset):
    """
    Implementation of a sequential dataset.

    pragma: skip_doc
    """
    def __init__(self, tens: torch.Tensor, sequence_length: int):
        """
        Args:
            tens (Tensor): Input tensor of shape [N, ...] containing the full sequence.
            sequence_length (int): Length of each input sequence window.
        """
        if tens.dim() == 1:
            self.data_tens = torch.unsqueeze(tens, -1)
        else:
            self.data_tens = tens

        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """
        Returns:
            int: Number of valid (input, target) sequence pairs.
        """
        return self.data_tens.size(0) - self.sequence_length - 1

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            i (int): Index of the sequence window.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - x: Input sequence of shape [sequence_length, ...]
                - y: Target sequence of shape [sequence_length, ...], shifted by one step
        """
        x = self.data_tens[i     : i + self.sequence_length    ]
        y = self.data_tens[i + 1 : i + self.sequence_length + 1]
        return x, y



class PtvSequentialTensorDataset:
    """
    Creates a sequential tensor Dataset.
        
        Args:  
            tens (torch.Tensor): The tensor containing the data.
            seq_len (int): Sequence length.

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
                "tens": ("TENSOR", {}),
                "seq_len": ("INT", {"default":32, "min":1, "max":1e5})
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTVDATASET",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self, 
          tens: torch.Tensor,
          seq_len: int) -> Tuple:
        """
        Creates a sequential tensor Dataset.
        
        Args:  
            tens (torch.Tensor): The tensor containing the data.
            seq_len (int): Sequence length.
            
        Returns:  
            Tuple: A tuple containing the dataset instance.  
        """
        
        with torch.inference_mode(False):
            dc = SequentialTensorDataset(tens, seq_len)
            return (dc,)
