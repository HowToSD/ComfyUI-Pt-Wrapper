from typing import Any, Dict, Tuple, Callable
import torch


class PtvHfDatasetWithTokenEncode:
    """
    A PyTorch Dataset class wrapper for a Hugging Face dataset that converts text to token IDs.

        Args:  
            name (str): Name of the dataset class.  
            split (str): The name of the subset of dataset to retrieve or the split specifier.
            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
            encode (PTCALLABLE): The reference to a token encoder function.

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
                "sample_field_name": ("STRING", {"default": "text","multiline": False}),
                "label_field_name": ("STRING", {"default": "label","multiline": False}),
                "encode": ("PTCALLABLE", {}),
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
          sample_field_name: str,
          label_field_name: str,
          encode: Callable) -> Tuple:
        """
        Loads a dataset from Hugging Face with specified parameters.  
        
        Args:  
            name (str): Name of the dataset class (e.g., MNIST, FashionMNIST).  
            split (str): The name of the subset of dataset to retrieve or the split specifier.
            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
            encode (Callable): The reference to a token encoder function.

        Returns:  
            Tuple: A tuple containing the dataset instance.  
        """
        # Keep import within this method so that relevant packages are imported
        # only when they are actually needed.
        from .hf_dataset_with_token_encode import HfDatasetWithTokenEncode

        with torch.inference_mode(False):
            dc = HfDatasetWithTokenEncode(
                dataset_name=name,
                split=split,
                sample_field_name=sample_field_name,
                label_field_name=label_field_name,
                encode=encode)
            return (dc,)
