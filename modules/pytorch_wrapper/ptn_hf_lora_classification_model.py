from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from .ptn_hf_lora_classification_model_def import HfLoraClassificationModel

class PtnHfLoraClassificationModel:
    """
    A binary classification model containing a Hugging Face pretrained Transformer model.

        Args:  
            model_name (str): Name or path of the pretrained Hugging Face model.  
            use_mean_pooling (bool): If True, average token embeddings using the attention mask.  
                                    Otherwise, use pooler output (if available) or the CLS token.  
            dropout (float): Dropout rate applied before the final classification layer.  
                            Set to 0.0 to disable dropout. Must be ≥ 0.  
            lora_r (int): LoRA rank parameter (dimension of the low-rank matrices).  
            lora_alpha (int): LoRA scaling factor. The adapted weight is added as:  
                            (lora_alpha / lora_r) * (A @ B), where  
                            A is a (m × r) matrix and B is a (r × n) matrix.  
            lora_dropout (float): Dropout applied to the input of the LoRA layers.  

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
                "dropout": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001}),
                "lora_r": ("INT", {"default": 8, "min": 1, "max": 1e6}),
                "lora_alpha": ("INT", {"default": 16, "min": 1, "max": 1e6}),
                "lora_dropout": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.001}),
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
          dropout: float,
          lora_r: int = 8,
          lora_alpha: int = 16,
          lora_dropout: float = 0.1
        ) -> Tuple[nn.Module]:
        """
        Instantiates a binary classification model containing a Hugging Face pretrained Transformer model.

            Args:
            model_name (str): Name or path of the pretrained Hugging Face model.
            use_mean_pooling (bool): If True, average token embeddings using the attention mask.
            Otherwise, use pooler output or CLS token embedding.
            dropout (float): Dropout rate applied before the final linear layer.
                             Set to 0.0 to disable dropout. Must be ≥ 0.    
            lora_r (int): LoRA rank parameter (dimension of the low-rank matrices).
            lora_alpha (int): LoRA scaling factor. The adapted weight is added as:
                            (lora_alpha / lora_r) * (A @ B), where
                            A is a (m × r) matrix and B is a (r × n) matrix.
            lora_dropout (float): Dropout applied to the input of the LoRA layers.

            Returns:
                Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
        with torch.inference_mode(False):
            model = HfLoraClassificationModel(
                    model_name=model_name,
                    use_mean_pooling=use_mean_pooling,
                    dropout=dropout,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )

        return (model,)
