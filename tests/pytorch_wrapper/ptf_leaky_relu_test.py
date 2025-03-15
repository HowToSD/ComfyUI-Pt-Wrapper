import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptf_leaky_relu import PtfLeakyReLU


class TestPtfRelu(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtfLeakyReLU()

        # Test cases: (Input Tensor, Expected LeakyReLU Output)
        self.test_cases = [
            (torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([-0.1, 0.0, 1.0])),
            (torch.tensor([[-1.0, 2.0], [3.0, -4.0]]), torch.tensor([[-0.1, 2.0], [3.0, -0.4]])),
            (torch.tensor([0.5, -0.5, 2.0, -2.0]), torch.tensor([0.5, -0.05, 2.0, -0.2])),
        ]

    def test_leaky_relu_function(self):
        """Test the returned LeakyReLU function."""
        leaky_relu_function, = self.node.f(negative_slope=0.1)

        # Ensure return value is callable
        self.assertTrue(callable(leaky_relu_function), "Returned value is not callable")

        for input_tensor, expected in self.test_cases:
            with self.subTest(input_tensor=input_tensor):
                result = leaky_relu_function(input_tensor)

                # Ensure output is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected output
                self.assertTrue(torch.equal(result, expected), f"Expected {expected}, got {result}")


if __name__ == "__main__":
    unittest.main()
