from typing import Any, Dict, Union, List, Tuple, Callable
import torch
from transformers import AutoTokenizer


class HfTokenizerEncode:
    """
    Hugging Face tokenizer wrapper for encoding text into token ID tensors.

        Args:  
            model_name (str): Name of the model.
            padding (bool): Whether to pad sequences to equal length.
            padding_method (str): 'longest' or 'max_length'.
                - 'longest': pad to the longest sequence length after truncation in this batch.
                - 'max_length': pad all sequences to the fixed length `max_length`.
            truncation (bool): Whether to truncate sequences to `max_length`.
            max_length (int): The target length for truncation or padding.  
                              If you specify 0, it uses model's maximum input length.

        Returns:  
            PTCALLABLE: A tuple containing a callable that accepts a sentence or list of
            sentences and returns token ID tensor(s) and attention mask(s).  

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
                "model_name": ("STRING", {"default":"", "multiline":False}),
                "padding": ("BOOLEAN", {"default": True}),
                "padding_method": (("max_length", "longest"),),
                "truncation": ("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 512, "min": 0, "max": 1_000_000}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTCALLABLE",)
    FUNCTION: str = "f"
    CATEGORY: str = "Data Analysis"

    def encode(
        self,
        sentence: Union[str, List[str]],
    ) -> Dict[str, torch.Tensor]:

        """
        Encodes a string or list of strings into token ID tensors.

        Args:
            sentence (Union[str, List[str]]): Input sentence(s).

        Returns:
            Dict:
             input_ids: token IDs
             attention_mask: Masks. 1 is non-padding, 0 is padding.
        """
        if self.padding:
            if self.max_length == 0:
                retval = self.tokenizer(
                            sentence,
                            padding=self.padding_method,
                            truncation=self.truncation,
                            return_tensors="pt"
                        )
            else:
                retval = self.tokenizer(
                            sentence,
                            padding=self.padding_method,
                            truncation=self.truncation,
                            max_length=self.max_length,
                            return_tensors="pt"
                        )
        else:  # no padding
            if self.max_length == 0:
                retval = self.tokenizer(
                            sentence,
                            padding=False,
                            truncation=self.truncation,
                            return_tensors="pt"
                        )
            else:
                retval = self.tokenizer(
                            sentence,
                            padding=False,
                            truncation=self.truncation,
                            return_tensors="pt"
                        )
        return retval

    def f(
        self,
        model_name: str,
        padding: bool,
        padding_method: str,
        truncation: bool,
        max_length: int
    ) -> Tuple[Callable[[Union[str, List[str]]], Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]]]:
        """
        Instantiates a Hugging Face tokenizer, binds configuration and returns a callable encoder function.
        - Supports includes:
        - BertTokenizer
        - RobertaTokenizer
        - DistilBertTokenizer
        - AlbertTokenizer
        - ElectraTokenizer

        Args:
            model_name (str): Name of the model.
            padding (bool): Whether to pad sequences to equal length.
            padding_method (str): 'longest' or 'max_length'.
                - 'longest': pad to the longest sequence length after truncation in this batch.
                - 'max_length': pad all sequences to the fixed length `max_length`.
            truncation (bool): Whether to truncate sequences to `max_length`.
            max_length (int): The target length for truncation or padding.
                              If you specify 0, it uses model's maximum input length.

        Returns:
            Tuple[Callable]: A tuple containing a callable for the tokenizer function.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.padding = padding
        self.padding_method = padding_method
        self.truncation = truncation
        self.max_length = max_length
        return (self.encode,)
