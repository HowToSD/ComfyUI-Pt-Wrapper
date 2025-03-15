import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptf_softplus import PtfSoftplus


class TestPtfSoftplus(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtfSoftplus()

        # Test cases: (Input Tensor, Expected Softplus Output)
        self.test_cases = [
            (torch.tensor([-1.0, 0.0, 1.0]), torch.nn.functional.softplus(torch.tensor([-1.0, 0.0, 1.0]))),
            (torch.tensor([[-2.0, 2.0], [3.0, -3.0]]), torch.nn.functional.softplus(torch.tensor([[-2.0, 2.0], [3.0, -3.0]]))),
            (torch.tensor([0.5, -0.5, 5.0, -5.0]), torch.nn.functional.softplus(torch.tensor([0.5, -0.5, 5.0, -5.0]))),
        ]

    def test_softplus_function(self):
        """Test the returned Softplus function."""
        softplus_function, = self.node.f()

        # Ensure return value is callable
        self.assertTrue(callable(softplus_function), "Returned value is not callable")

        for input_tensor, expected in self.test_cases:
            with self.subTest(input_tensor=input_tensor):
                result = softplus_function(input_tensor)

                # Ensure output is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected output (allowing for floating-point tolerance)
                self.assertTrue(torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}")


if __name__ == "__main__":
    unittest.main()
