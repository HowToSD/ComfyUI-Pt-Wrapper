import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptf_relu import PtfReLU


class TestPtfRelu(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtfReLU()

        # Test cases: (Input Tensor, Expected ReLU Output)
        self.test_cases = [
            (torch.tensor([-1, 0, 1]), torch.tensor([0, 0, 1])),
            (torch.tensor([[-1, 2], [3, -4]]), torch.tensor([[0, 2], [3, 0]])),
            (torch.tensor([0.5, -0.5, 2.0, -2.0]), torch.tensor([0.5, 0, 2.0, 0])),
        ]

    def test_relu_function(self):
        """Test the returned ReLU function."""
        relu_function, = self.node.f()

        # Ensure return value is callable
        self.assertTrue(callable(relu_function), "Returned value is not callable")

        for input_tensor, expected in self.test_cases:
            with self.subTest(input_tensor=input_tensor):
                result = relu_function(input_tensor)

                # Ensure output is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected output
                self.assertTrue(torch.equal(result, expected), f"Expected {expected}, got {result}")


if __name__ == "__main__":
    unittest.main()
