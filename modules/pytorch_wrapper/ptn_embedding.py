from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class PtnEmbedding:
    """
    Constructs an embedding layer.

        Args:
            num_embeddings (int): Vocabulary size.  
            embedding_dim (int): The size of each embedding vector.  
            padding_idx (int): Token ID to treat as padding. If a non-negative integer is specified, the embedding at that index will be excluded from gradient updates. Set to -1 to disable padding behavior (default).

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
                "num_embeddings": ("INT", {"default": 10000, "min": 1, "max": 1e6}),
                "embedding_size": ("INT", {"default": 100, "min": 1, "max": 1e6}),
                "padding_idx": ("INT", {"default": -1, "min": -1, "max": 1e6}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self,
          num_embeddings: int,
          embedding_dim: int,
          padding_idx: int
          ) -> Tuple[nn.Module]:
        """
        Constructs an embedding layer.

        Args:
            num_embeddings (int): Vocabulary size.  
            embedding_dim (int): The size of each embedding vector.  
            padding_idx (int): Token ID to treat as padding. If a non-negative integer is specified, the embedding at that index will be excluded from gradient updates. Set to -1 to disable padding behavior (default).

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
 
        with torch.inference_mode(False):
            model = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx)

        return (model,)
