import os
import sys
from typing import Any, Dict
import torch
import sentencepiece as spm

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.utils import get_model_full_path

class SpLoadModel:
    """
    A wrapper class for loading a SentencePiece tokenization model.

    category: PyTorch wrapper - Training
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types for the function.

        Returns:
            Dict[str, Any]: A dictionary specifying required input types.
        """
        return {
            "required": {
                "model_path": ("STRING", {"default": "spiece.model"})
            }
        }

    RETURN_NAMES: tuple = ("spmodel",)
    RETURN_TYPES: tuple = ("SPMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Data Analysis"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self, model_path: str):
        model_full_path = get_model_full_path(model_path,
                                              sub_model_dir_name="sentencepiece")
        sp = spm.SentencePieceProcessor(model_file=model_full_path)

        return (sp,)