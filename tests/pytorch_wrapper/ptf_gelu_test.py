import os
import sys
import unittest
import torch
import torch.nn.functional

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptf_gelu import PtfGELU


class TestPtfGELU(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtfGELU()

        # Test cases: (Input Tensor, Expected GELU Output)
        self.test_cases = [
            (torch.tensor([-1.0, 0.0, 1.0]), torch.nn.functional.gelu(torch.tensor([-1.0, 0.0, 1.0]))),
            (torch.tensor([[-1.0, 2.0], [3.0, -4.0]]), torch.nn.functional.gelu(torch.tensor([[-1.0, 2.0], [3.0, -4.0]]))),
            (torch.tensor([0.5, -0.5, 2.0, -2.0]), torch.nn.functional.gelu(torch.tensor([0.5, -0.5, 2.0, -2.0]))),
        ]

    def test_gelu_function(self):
        """Test the returned GELU function."""
        gelu_function, = self.node.f()

        # Ensure return value is callable
        self.assertTrue(callable(gelu_function), "Returned value is not callable")

        for input_tensor, expected in self.test_cases:
            with self.subTest(input_tensor=input_tensor):
                result = gelu_function(input_tensor)

                # Ensure output is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected output
                self.assertTrue(torch.equal(result, expected), f"Expected {expected}, got {result}")


if __name__ == "__main__":
    unittest.main()
