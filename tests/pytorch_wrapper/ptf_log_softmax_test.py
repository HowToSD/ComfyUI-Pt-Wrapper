import os
import sys
import unittest
import torch
import torch.nn.functional

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptf_log_softmax import PtfLogSoftmax


def compute_log_softmax(x, dim=-1):
    return torch.nn.functional.log_softmax(x, dim=dim)


class TestPtfLogSoftmax(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtfLogSoftmax()

        # Test cases: (Input Tensor, Expected LogSoftmax Output)
        self.test_cases = [
            # Rank-1 tensor
            (torch.tensor([0.0, -100.0, 100.0]), compute_log_softmax(torch.tensor([0.0, -100.0, 100.0]))),
            # # Rank-2 tensor dim=-1
            (torch.tensor([[-1.0, 2.0, 3.0], [0.0, 1.0, -1.0]]), compute_log_softmax(torch.tensor([[-1.0, 2.0, 3.0], [0.0, 1.0, -1.0]]))),
            # # Rank-2 tensor dim=1
            (torch.tensor([[-1.0, 2.0, 3.0], [0.0, 1.0, -1.0]]), compute_log_softmax(torch.tensor([[-1.0, 2.0, 3.0], [0.0, 1.0, -1.0]]), dim=1)),
            # # Rank-3 tensor
            (torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.0, -1.0], [-2.0, -3.0]]]), compute_log_softmax(torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.0, -1.0], [-2.0, -3.0]]]))),
        ]

    def test_log_softmax_function(self):
        """Test the returned LogSoftmax function."""
        log_softmax_function, = self.node.f()

        # Ensure return value is callable
        self.assertTrue(callable(log_softmax_function), "Returned value is not callable")

        for input_tensor, expected in self.test_cases:
            with self.subTest(input_tensor=input_tensor):
                result = log_softmax_function(input_tensor)

                # Ensure output is a tensor
                self.assertTrue(isinstance(result, torch.Tensor), "Output is not a tensor")

                # Ensure result matches expected output
                self.assertTrue(torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}")


if __name__ == "__main__":
    unittest.main()
