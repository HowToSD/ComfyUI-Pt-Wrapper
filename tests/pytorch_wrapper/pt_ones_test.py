import os
import sys
import unittest
import torch


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pt_ones import PtOnes  # Importing the PtOnes class
from pytorch_wrapper.utils import DTYPE_MAPPING  # Importing the dtype mapping


class TestPtOnes(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtOnes()

        # Tensor size to test
        self.size_str = "[2, 3]"  # Using a small 2D tensor size for all tests
        self.expected_shape = (2, 3)

    def test_ones_tensor_all_dtypes(self):
        """Test tensor creation with ones for all supported data types."""
        for data_type, torch_dtype in DTYPE_MAPPING.items():
            with self.subTest(data_type=data_type):
                result, = self.node.f(self.size_str, data_type)

                # Ensure return value is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result has correct shape
                self.assertEqual(result.shape, torch.Size(self.expected_shape), f"Expected shape {self.expected_shape}, got {result.shape}")

                # Ensure result has correct dtype
                self.assertEqual(result.dtype, torch_dtype, f"Expected dtype {torch_dtype}, got {result.dtype}")

                # Ensure all elements are ones
                expected = torch.ones(self.expected_shape, dtype=torch_dtype)
                self.assertTrue(torch.equal(result, expected), f"Tensor values do not match expected ones tensor for dtype {data_type}")

if __name__ == "__main__":
    unittest.main()
