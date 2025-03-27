from typing import Any, Dict, Callable, Tuple, List, Union


class PtTokenizer:
    """
    The tokenizer to encode a string or a list of string to token IDs.

        Args:  
            encode (PTCALLABLE): The reference to a token encoder function.  
            text_list (PYLIST): The python list of strings to tokenize.  

        Returns:  
            A tuple containing the token IDs and the mask in torch.Tensor.  

    category: PyTorch wrapper - Callable
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the required input types.

        Returns:
            Dict[str, Any]: An empty dictionary since this function requires no inputs.
        """
        return {
            "required": {
                "encode": ("PTCALLABLE", {}),
                "text_list": ("PYLIST", {})
            }
        }

    RETURN_NAMES: Tuple[str, ...] = ("token_tens", "mask_tens")
    RETURN_TYPES: Tuple[str, ...] = ("TENSOR", "TENSOR")
    FUNCTION: str = "f"
    CATEGORY: str = "Data Analysis"

    def f(self,
          encode: Callable,
          text_list: Union[str, List[str]]) -> Tuple[Callable]:
        """
        Args:
            encode (Callable): The reference to a token encoder function.
            text_list (Union[str, List[str]]): The a string or a python list of strings to tokenize.

        Returns:
            Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]: A tuple containing the token IDs and the mask in torch.Tensor.
        """
        return encode(text_list)
