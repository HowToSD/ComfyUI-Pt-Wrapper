import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pt_apply_function import PtApplyFunction


class TestPtApplyFunction(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtApplyFunction()

        # Softplus function for testing
        self.test_function = torch.nn.functional.softplus

        # Test cases: (Input Tensor, Expected Softplus Output)
        self.test_cases = [
            (torch.tensor([-1.0, 0.0, 1.0]), self.test_function(torch.tensor([-1.0, 0.0, 1.0]))),
            (torch.tensor([[-2.0, 2.0], [3.0, -3.0]]), self.test_function(torch.tensor([[-2.0, 2.0], [3.0, -3.0]]))),
            (torch.tensor([0.5, -0.5, 5.0, -5.0]), self.test_function(torch.tensor([0.5, -0.5, 5.0, -5.0]))),
        ]

    def test_apply_function(self):
        """Test the function application."""
        for input_tensor, expected in self.test_cases:
            with self.subTest(input_tensor=input_tensor):
                result, = self.node.f(input_tensor, self.test_function)

                # Ensure output is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected output (allowing for floating-point tolerance)
                self.assertTrue(torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}")


if __name__ == "__main__":
    unittest.main()
