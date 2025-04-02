import os
import sys
from typing import Any, Dict, Tuple, Callable
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from hugging_face_wrapper.hf_local_dataset import HfLocalDataset

class PtvHfLocalDataset:
    """
    A PyTorch Dataset class wrapper for a Hugging Face dataset to load a dataset that is stored on the local file system.
    
        Args:  
            file_path (str): Path to the dataset file. Can be an absolute path or a relative path.  
            If a relative path is provided, it is resolved from the `datasets` directory under this extension.  
            For example, to load `datasets/foo/bar_train.jsonp`, specify `foo/bar_train.jsonp` or the absolute path.
            file_format (str): File format (e.g. json if the file is in jsonl format).  
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
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "file_format": ("STRING", {"default": "json","multiline": False}),
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
          file_path: str,
          file_format: str,
          sample_field_name: str,
          label_field_name: str,
          encode: Callable,
          remove_html_tags:bool,
          encode_return_dict:bool) -> Tuple:
        """
        Loads a dataset from Hugging Face with specified parameters.  
        
        Args:  
            file_path (str): Path to the dataset file. Can be an absolute path or a relative path.  
            If a relative path is provided, it is resolved from the `datasets` directory under this extension.  
            For example, to load `datasets/foo/bar_train.jsonp`, specify `foo/bar_train.jsonp` or the absolute path.
            file_format (str): File format (e.g. json if the file is in jsonl format).            sample_field_name (str): Field name for text samples.
            label_field_name (str): Field name for labels.
            encode (Callable): The reference to a token encoder function.
            remove_html_tags (bool): Remove html tags in text if True.
            encode_return_dict (bool): Encode function returns a Dict instead of a Tuple.

        Returns:  
            Tuple: A tuple containing the dataset instance.  
        """
        with torch.inference_mode(False):
            dc = HfLocalDataset(
                file_path=file_path,
                file_format=file_format,
                sample_field_name=sample_field_name,
                label_field_name=label_field_name,
                encode=encode,
                remove_html_tags=remove_html_tags,
                encode_return_dict=encode_return_dict)
            return (dc,)
