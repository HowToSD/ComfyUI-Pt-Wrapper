import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pt_stack import PtStack


class TestPtStack(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtStack()

        # Test cases: (Tensor A, Tensor B, Dim, Expected result)
        self.test_cases = [
            (torch.tensor([1, 2]), torch.tensor([5, 6]), 0, torch.tensor([[1, 2], [5, 6]])),
            (torch.tensor([1, 2]), torch.tensor([5, 6]), 1, torch.tensor([[1, 5], [2, 6]]))
        ]

    def test_stacking(self):
        """Test tensor stacking along different dimensions."""
        for tens_a, tens_b, dim, expected in self.test_cases:
            with self.subTest(tens_a=tens_a, tens_b=tens_b, dim=dim):
                result, = self.node.f(tens_a, tens_b, dim)

                # Ensure return value is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected stacking
                self.assertTrue(torch.equal(result, expected), f"Expected {expected}, got {result}")

if __name__ == "__main__":
    unittest.main()