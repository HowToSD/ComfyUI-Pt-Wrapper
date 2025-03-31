import os
import sys
from typing import Any, Dict, Tuple, Callable
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from hugging_face_wrapper.hf_dataset_with_token_encode import HfDatasetWithTokenEncode

class PtvHfDatasetWithTokenEncode:
    """
    A PyTorch Dataset class wrapper for a Hugging Face dataset that converts text to token IDs.

        Args:  
            name (str): Name of the dataset class.  
            split (str): The name of the subset of dataset to retrieve or the split specifier.
            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
            encode (PTCALLABLE): The reference to a token encoder function.
            remove_html_tags (bool): Remove html tags in text if True.
            encode_return_dict (bool): Encode function returns a Dict instead of a Tuple.

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
                "remove_html_tags": ("BOOLEAN", {"default": False}),
                "encode_return_dict": ("BOOLEAN", {"default": False}),
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
          encode: Callable,
          remove_html_tags:bool,
          encode_return_dict:bool) -> Tuple:
        """
        Loads a dataset from Hugging Face with specified parameters.  
        
        Args:  
            name (str): Name of the dataset class (e.g., MNIST, FashionMNIST).  
            split (str): The name of the subset of dataset to retrieve or the split specifier.
            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
            encode (Callable): The reference to a token encoder function.
            remove_html_tags (bool): Remove html tags in text if True.
            encode_return_dict (bool): Encode function returns a Dict instead of a Tuple.

        Returns:  
            Tuple: A tuple containing the dataset instance.  
        """
        with torch.inference_mode(False):
            dc = HfDatasetWithTokenEncode(
                dataset_name=name,
                split=split,
                sample_field_name=sample_field_name,
                label_field_name=label_field_name,
                encode=encode,
                remove_html_tags=remove_html_tags,
                encode_return_dict=encode_return_dict)
            return (dc,)
