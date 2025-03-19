import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pt_concat import PtConcat


class TestPtConcat(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtConcat()

        # Test cases: (Tensor A, Tensor B, Dim, Expected result)
        self.test_cases = [
            (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]), 0, torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])),
            (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]), 1, torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]])),
            (torch.ones(1, 2, 2), torch.zeros(1, 2, 2), 2, torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0]]])),
            (torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), 3, torch.tensor([[[[1, 1, 0, 0], [1, 1, 0, 0]]]])),
        ]

    def test_concatenation(self):
        """Test tensor concatenation along different dimensions."""
        for tens_a, tens_b, dim, expected in self.test_cases:
            with self.subTest(tens_a=tens_a, tens_b=tens_b, dim=dim):
                result, = self.node.f(tens_a, tens_b, dim)

                # Ensure return value is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected concatenation
                self.assertTrue(torch.equal(result, expected), f"Expected {expected}, got {result}")

if __name__ == "__main__":
    unittest.main()
