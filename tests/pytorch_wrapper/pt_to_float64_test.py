import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pt_to_float64 import PtToFloat64


class TestPtToFloat64(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtToFloat64()

        # Test cases: (Tensor A, expected result, expected dtype)
        self.test_cases = [
            (torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), torch.float64),
        ]

    def test_1(self):
        """Test tensor element-wise square root using torch.sqrt()."""
        for tens_a, expected_value, expected_dtype in self.test_cases:
            with self.subTest(tens_a=tens_a):
                result, = self.node.f(tens_a)

                # Ensure return value is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure data type
                self.assertEqual(result.dtype, expected_dtype,
                                 f"Expected dtype {expected_dtype}, got {result.dtype}")

                # Ensure result matches expected values
                self.assertTrue(torch.allclose(result, expected_value, atol=1e-6), f"Expected {expected_value}, got {result}")


if __name__ == "__main__":
    unittest.main()
