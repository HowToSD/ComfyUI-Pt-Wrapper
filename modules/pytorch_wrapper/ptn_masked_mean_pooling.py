from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

class PtnMaskedMeanPoolingDef(nn.Module):
    """
    Constructs a masked mean pooling layer.

    pragma: skip_doc
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean of the input excluding the padding values using the given mask.

        Args:
            inputs (Tensor): 3D input tensor in (Batch, Sequence, Features) format.
            mask (Tensor): 2D input tensor in (Batch, Sequence) format. Padding is indicated by 0; valid tokens are indicated by 1.

        Returns:
            torch.Tensor: Output tensor.
        """
        if mask.dim() != 2:
            raise ValueError("Mask is not rank 2")
        if inputs.dim() != 3:
            raise ValueError("Inputs is not rank 2")
        
        maskf = mask.float()  # (B, Seq)
        x = (inputs * maskf.unsqueeze(-1)).sum(dim=1) / maskf.sum(dim=1, keepdim=True).clamp(min=1e-9)  # (B, Feature)
        if x.dim() != 2:
            raise ValueError("Outputs is not rank 2")
        return x


class PtnMaskedMeanPooling:
    """
    Constructs a masked mean pooling layer.
    Computes the mean of the input excluding the padding values using the given mask.

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
        }

    RETURN_TYPES: Tuple[str] = ("PTMODEL",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("NaN")

    def f(self) -> Tuple[nn.Module]:
        """
        Constructs a masked mean pooling layer.
        Computes the mean of the input excluding the padding values using the given mask.

        Returns:
            Tuple[nn.Module]: A tuple containing the instantiated PyTorch model.
        """
 
        with torch.inference_mode(False):
            model = PtnMaskedMeanPoolingDef()

        return (model,)
