"""
Util functions for unit tests.
"""
import torch
import numpy as np
from typing import Union, List
import unittest

def assert_equal_for_tensor_and_list_or_scalar(
    test_inst: unittest.TestCase,
    actual: torch.Tensor,
    expected: Union[List[float], float, np.ndarray],
    message: str = ""
) -> None:
    """
    Asserts that a given tensor's value is approximately equal to the expected value.
    
    This function supports comparison between a PyTorch tensor and:
    - A scalar float
    - A list of floats
    - A NumPy array

    Args:
        test_inst (unittest.TestCase): The test instance to perform assertions.
        actual (torch.Tensor): The tensor containing computed values.
        expected (Union[List[float], float, np.ndarray]): The expected values.
        message (str): Additional message to include in assertion errors.

    Raises:
        ValueError: If `expected` is not a float, list of floats, or a NumPy array.
    """
    if isinstance(expected, float):
        # Convert tensor to scalar float
        actual_value = actual.item()
        test_inst.assertAlmostEqual(
            actual_value, expected, places=4, 
            msg=f"{message}: Expected {expected}, got {actual_value}"
        )

    elif isinstance(expected, list) or isinstance(expected, np.ndarray):
        # Convert tensor to numpy array for comparison
        actual_array = actual.cpu().numpy()
        expected_array = np.array(expected, dtype=actual_array.dtype)

        test_inst.assertTrue(
            np.allclose(actual_array, expected_array, atol=1e-5),
            msg=f"{message}: Expected {expected_array}, got {actual_array}"
        )

    else:
        raise ValueError(
            f"Unsupported data type for `expected`: {type(expected)}"
        )
