import os
import ast
from typing import Optional, Tuple, Union, List
import random
import numpy as np
import torch
import torch.nn as nn

DTYPE_MAPPING = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool
}

def str_to_number(s: str) -> Union[int, float]:
    """
    Converts a string representation of a number to an integer or float.

    Args:
        s (str): The string to convert.

    Returns:
        Union[int, float]: The converted number, either as an `int` if no decimal point is present or as a `float` otherwise.

    Raises:
        ValueError: If the input string is empty.
    """
    if s:
        if "." in s:
            return float(s)
        return int(s)
    raise ValueError("Input string cannot be empty.")


def str_to_number_with_default(s: str, n: Union[int, float]) -> Union[int, float]:
    """
    Converts a string representation of a number to an integer or float, with a fallback default value.

    Args:
        s (str): The string to convert.
        n (Union[int, float]): The default value to return if `s` is empty.

    Returns:
        Union[int, float]: The converted number if `s` is non-empty, otherwise `n`.
    """
    return str_to_number(s) if s else n

def get_resource_full_path(resource_name: str, resource_path: str) -> str:
    """
    Resolves the absolute path to a resource file within the "resources" directory and 
    ensures that the directory exists.

    Args:
        resource_path (str): The path to the resource file. If the path is a relative path,
        the path is considered relative to the `resource` directory of this extension.

    Returns:
        str: The absolute path to the resource file.
    """
    if os.path.isabs(resource_path):
        resource_full_path = resource_path
    else:
        resource_full_path = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "..", resource_name, resource_path))
        resource_dir = os.path.dirname(resource_full_path)
        if os.path.exists(resource_dir) is False:
            os.makedirs(resource_dir)
    return resource_full_path


def get_model_full_path(model_path: str, sub_model_dir_name: str=None) -> str:
    """
    Resolves the absolute path to a model file within the "models" directory and 
    ensures that the directory exists.

    Args:
        model_path (str): The path to the model file. If the path is a relative path,
        the path is considered relative to the `models` directory of this extension.
        sub_model_dir_name (str): The subdirectory specific to the type of model (e.g. SentencePiece).
    Returns:
        str: The absolute path to the model file.
    """
    if sub_model_dir_name:
        model_dir = os.path.join("models", sub_model_dir_name)
    else:
        model_dir = "models"
    return get_resource_full_path(model_dir, model_path)


def get_dataset_full_path(dataset_path: str) -> str:
    """
    Resolves the absolute path to the datasete within the "datasets" directory and 
    ensures that the directory exists.

    Args:
        dataset_path (str): The path to the dataset file. If the path is a relative path,
        the path is considered relative to the `datasets` directory of this extension.

    Returns:
        str: The absolute path to the dataset file.
    """
    return get_resource_full_path("datasets", dataset_path)


def str_to_dim(dim: str) -> Optional[Union[int, Tuple[int, ...]]]:
    """
    Converts a string representation of a dimension into an integer or a tuple of integers.

    Args:
        dim (str): A string representing an integer or a tuple of integers.

    Returns:
        Optional[Union[int, Tuple[int, ...]]]: 
            - An integer if the input represents a single integer.
            - A tuple of integers if the input represents a list or tuple.
            - None if the input is empty.

    Raises:
        ValueError: If the parsed value is not an integer or a tuple of integers.
    """
    if dim:
        # Convert string to an integer, list, or tuple of integers
        parsed_dim = ast.literal_eval(dim)

        # Convert list to tuple
        if isinstance(parsed_dim, list):
            parsed_dim = tuple(parsed_dim)

        # Ensure the parsed dim is valid
        if not isinstance(parsed_dim, (int, tuple)) or (
            isinstance(parsed_dim, tuple) and not all(isinstance(i, int) for i in parsed_dim)
        ):
            raise ValueError(f"Invalid format for dim: {parsed_dim}. Must be an integer or a tuple of integers.")
        
        return parsed_dim

    return None

def set_seed(seed=42) -> None:
    """
    Sets seed for random number generation.

    Args:
        seed (int): Initial seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pad_truncate_sequence(
    token_tensor_list: List[torch.Tensor],
    padding: bool,
    padding_method: str,
    padding_value: Union[int, torch.Tensor],
    truncation: bool,
    max_length: int
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[List[torch.Tensor], List[torch.Tensor]]
]:
    """
    Pads and/or truncates a list of 1D token tensors to a uniform length.

    Args:
        token_tensor_list (List[torch.Tensor]): List of 1D token tensors.
        padding (bool): Whether to apply padding.
        padding_method (str): 'longest' or 'max_length'.
            - 'longest': pad to the longest sequence length after truncation in this batch.
            - 'max_length': pad all sequences to the fixed length `max_length`.
        padding_value (Union[int, torch.Tensor]): A scalar value or scalar tensor used for padding.
        truncation (bool): Whether to truncate sequences to `max_length`.
        max_length (int): The target length for truncation or padding.

    Returns:
        Union[
            Tuple[torch.Tensor, torch.Tensor],  # (tokens, masks) if padding is True or list has 1 element
            Tuple[List[torch.Tensor], List[torch.Tensor]]  # (tokens, masks) as lists otherwise
        ]
    """
    assert isinstance(token_tensor_list, list) and all(isinstance(t, torch.Tensor) for t in token_tensor_list), \
        "token_tensor_list must be a list of torch.Tensor"
    if isinstance(padding_value, torch.Tensor):
        assert padding_value.numel() == 1, "padding_value must be a scalar tensor"
        padding_value = padding_value.item()

    if padding_method not in ("max_length", "longest"):
        raise ValueError("Invalid padding_method. Use 'max_length' or 'longest'.")

    token_list = []
    mask_list = []

    # Apply truncation and collect post-truncation lengths
    truncated_list = []
    for t in token_tensor_list:
        if t.dim() != 1:
            raise ValueError("Expecting 1D tensor")
        if truncation:
            t = t[:max_length]  # Out of bounds check is not needed as PyTorch handles that.
        truncated_list.append(t)

    if padding and padding_method == "longest":
        padding_target_length = max(t.size(0) for t in truncated_list)
    elif padding and padding_method == "max_length":
        padding_target_length = max_length

    for t in truncated_list:
        tensor_length = t.size(0)

        # Apply padding if needed
        if padding and tensor_length < padding_target_length:
            filler_count = padding_target_length - tensor_length
            pad_tensor = torch.full((filler_count,), padding_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad_tensor], dim=0)

        # Create attention mask
        if padding:
            mask = (t != padding_value).long()
        else:
            mask = torch.ones_like(t)

        token_list.append(t)
        mask_list.append(mask)

    if padding or len(token_tensor_list) == 1:
        return torch.stack(token_list), torch.stack(mask_list)
    else:
        return token_list, mask_list


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Enables or disables gradient computation for all parameters in a given module.

    Args:
        module (nn.Module): The PyTorch module whose parameters should be modified.
        requires_grad (bool): If False, freezes the module (no gradients computed).
                              If True, unfreezes the module (gradients computed).
    """
    for param in module.parameters():
        param.requires_grad = requires_grad
