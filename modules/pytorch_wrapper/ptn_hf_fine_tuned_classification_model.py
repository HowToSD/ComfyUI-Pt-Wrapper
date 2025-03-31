from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from .ptn_hf_fine_tuned_classification_model_def import HfFineTunedClassificationModel

class PtnHfFineTunedClassificationModel:
    """
    A binary classification model containing a Hugging Face pretrained Transformer model.

            Args:
            model_name (str): Name or path of the pretrained Hugging Face model.
            use_mean_pooling (bool): If True, average token embeddings using the attention mask.
            Otherwise, use pooler output or CLS token embedding.
            dropout (float): Dropout rate applied before the final linear layer.
                             Set to 0.0 to disable dropout. Must be ≥ 0. 

    category: PyTorch wrapper - Model
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
                "model_name": ("STRING", {"default": "", "multiline": False}),
                "use_mean_pooling": ("BOOLEAN", {"default": True}),
                "dropout": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001})
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          model_name: str,
          use_mean_pooling: bool,
          dropout: float
        ) -> Tuple[nn.Module]:
        """
        Instantiates a binary classification model containing a Hugging Face pretrained Transformer model.

            Args:
            model_name (str): Name or path of the pretrained Hugging Face model.
            use_mean_pooling (bool): If True, average token embeddings using the attention mask.
            Otherwise, use pooler output or CLS token embedding.
            dropout (float): Dropout rate applied before the final linear layer.
                             Set to 0.0 to disable dropout. Must be ≥ 0.    
                                
            Returns:
                Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = HfFineTunedClassificationModel(
                    model_name=model_name,
                    use_mean_pooling=use_mean_pooling,
                    dropout=dropout
                )

        return (model,)
