import os
import sys
from typing import Any, Dict, Union, List, Tuple, Callable
import torch
import sentencepiece as spm

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.utils import pad_truncate_sequence


class SpEncode:
    """
    SentencePiece wrapper for encoding text into token ID tensors.

        Args:  
            spmodel (SentencePieceProcessor): Trained SentencePiece model.  
            padding (bool): Whether to pad sequences to equal length.  
            padding_method (str): 'longest' or 'max_length'.  
                - 'longest': pad to the longest sequence length after truncation in this batch.  
                - 'max_length': pad all sequences to the fixed length `max_length`.  
            padding_value (Union[int, torch.Tensor]): A scalar value or scalar tensor used for padding.  
            truncation (bool): Whether to truncate sequences to `max_length`.  
            max_length (int): The target length for truncation or padding.  

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
                "spmodel": ("SPMODEL", {}),
                "padding": ("BOOLEAN", {"default": False}),
                "padding_method": (("max_length", "longest"),),
                "padding_value": ("INT", {"default": 0, "min":0, "max":255}),
                "truncation": ("BOOLEAN", {"default": False}),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 1_000_000}),
            }
        }

    RETURN_TYPES: Tuple[str] = ("PTCALLABLE",)
    FUNCTION: str = "f"
    CATEGORY: str = "Data Analysis"

    def encode(
        self,
        sentence: Union[str, List[str]],
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Encodes a string or list of strings into token ID tensors.

        Args:
            sentence (Union[str, List[str]]): Input sentence(s).

        Returns:
            Tuple of token tensors and attention masks:
                - If padding is True:
                    (2D token tensor, 2D attention mask tensor)
                - If padding is False:
                    - For a single string: (1D token tensor, 1D mask tensor)
                    - For a list: (List of 1D token tensors, List of 1D mask tensors)
        """
        if isinstance(sentence, str):
            rank1_input = True
            sentence = [sentence]
        else:
            rank1_input = False

        token_list = []
        for s in sentence:
            ids = self.spmodel.encode(s, out_type=int)
            if self.truncation and len(ids) > self.max_length:
                ids = ids[:self.max_length]
            token_list.append(torch.tensor(ids, dtype=torch.long))

        tokens, masks = pad_truncate_sequence(
            token_list,
            self.padding,
            self.padding_method,
            self.padding_value,
            self.truncation,
            self.max_length
        )

        if rank1_input:
            return tokens[0], masks[0]
        else:
            return tokens, masks

    def f(
        self,
        spmodel: spm.SentencePieceProcessor,
        padding: bool,
        padding_method: str,
        padding_value: int,
        truncation: bool,
        max_length: int
    ) -> Tuple[Callable[[Union[str, List[str]]], Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]]]:
        """
        Binds configuration and returns a callable encoder function.

        Args:
            spmodel (SentencePieceProcessor): Trained SentencePiece model.
            padding (bool): Whether to pad sequences to equal length.
            padding_method (str): 'longest' or 'max_length'.
                - 'longest': pad to the longest sequence length after truncation in this batch.
                - 'max_length': pad all sequences to the fixed length `max_length`.
            padding_value (Union[int, torch.Tensor]): A scalar value or scalar tensor used for padding.
            truncation (bool): Whether to truncate sequences to `max_length`.
            max_length (int): The target length for truncation or padding.

        Returns:
            Tuple[Callable]: A tuple containing a callable that accepts a sentence or list of
            sentences and returns token ID tensor(s) and attention mask(s).
        """
        self.spmodel = spmodel
        self.padding = padding
        self.padding_method = padding_method
        self.padding_value = padding_value
        self.truncation = truncation
        self.max_length = max_length
        return (self.encode,)
