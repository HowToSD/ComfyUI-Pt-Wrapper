from typing import Any, Dict
import torch


class PtdmPmfTensor:
    """
    Computes the probability for the input distribution.  This nodes accepts a tensor so it can be used to compute pmf for multiple values contained in a tensor.

    Args:
            distribution (torch.distributions.distribution.Distribution): Distribution.
            tens (torch.Tensor): k(s) in Tensor.

    category: PyTorch wrapper - Distribution
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types for the function.

        Returns:
            Dict[str, Any]: A dictionary specifying the required input types.
        """
        return {
            "required": {
                "distribution": ("PTDISTRIBUTION", {}),
                "tens": ("TENSOR", {})
            }
        }

    RETURN_TYPES: tuple = ("TENSOR",)
    FUNCTION: str = "f"
    CATEGORY: str = "Distribution"

    def f(self, distribution: torch.distributions.Distribution, tens: torch.Tensor) -> tuple:
        """
        Computes the probability for the input distribution.

        Args:
            distribution (torch.distributions.Distribution): Distribution.
            tens (torch.Tensor): Value(s) in Tensor.

        Returns:
            tuple containing the probability
        """
        return (torch.exp(distribution.log_prob(tens)),)
