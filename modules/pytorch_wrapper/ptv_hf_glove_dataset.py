from typing import Any, Dict, Tuple
import torch


class PtvHfGloveDataset:
    """
    A PyTorch Dataset class wrapper for a Hugging Face dataset that converts text to embedding using Glove.

        Args:  
            name (str): Name of the dataset class (e.g., MNIST, FashionMNIST).  
            split (str): The name of the subset of dataset to retrieve or the split specifier.
            embed_dim (int): Embedding dimension for Glove.
            max_seq_len (int): Sequence length.
            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.

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
                "name": ("STRING", {"default": "", "multiline": False}),
                "split": ("STRING", {"default": "train","multiline": False}),
                "embed_dim": ("INT", {"default": 100, "min":32, "max": 4096}),
                "max_seq_len": ("INT", {"default": 256, "min":8, "max": 4096}),
                "sample_field_name": ("STRING", {"default": "text","multiline": False}),
                "label_field_name": ("STRING", {"default": "label","multiline": False}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTVDATASET",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self, 
          name: str,
          split: str,
          embed_dim: int,
          max_seq_len: int,
          sample_field_name: str,
          label_field_name: str) -> Tuple:
        """
        Loads a dataset from Hugging Face with specified parameters.  
        
        Args:  
            name (str): Name of the dataset class (e.g., MNIST, FashionMNIST).  
            split (str): The name of the subset of dataset to retrieve or the split specifier.
            embed_dim (int): Embedding dimension for Glove.
            max_seq_len (int): Sequence length.
            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.

        Returns:  
            Tuple: A tuple containing the dataset instance.  
        """
        # Keep import within this method so that relevant packages are imported
        # only when they are actually needed.
        from .hf_glove_dataset import HfGloveDataset

        with torch.inference_mode(False):
            dc = HfGloveDataset(
                dataset_name=name,
                split=split,
                glove_dim=embed_dim,
                max_seq_len=max_seq_len,
                sample_field_name=sample_field_name,
                label_field_name=label_field_name)
            return (dc,)
