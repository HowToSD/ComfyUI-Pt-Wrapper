import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptf_tanh import PtfTanh


class TestPtfTanh(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtfTanh()

        # Test cases: (Input Tensor, Expected Tanh Output)
        self.test_cases = [
            (torch.tensor([0, -100.0, 100.0]), torch.tensor([0.0, -1.0, 1.0])),
            (torch.tensor([[-1000000.0, 200.0], [3000.0, -400.0]]), torch.tensor([[-1.0, 1.0], [1.0, -1.0]])),
            (torch.tensor([0.0, -100.0, 200.0, -20.0]), torch.tensor([0.0, -1.0, 1.0, -1.0])),
        ]

    def test_tanh_function(self):
        """Test the returned Tanh function."""
        tanh_function, = self.node.f()

        # Ensure return value is callable
        self.assertTrue(callable(tanh_function), "Returned value is not callable")

        for input_tensor, expected in self.test_cases:
            with self.subTest(input_tensor=input_tensor):
                result = tanh_function(input_tensor)

                # Ensure output is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected output
                self.assertTrue(torch.allclose(result, expected, atol=1e6), f"Expected {expected}, got {result}")


if __name__ == "__main__":
    unittest.main()
