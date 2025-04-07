import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_masked_mean_pooling import PtnMaskedMeanPooling


class TestPtnMaskedMeanPooling(unittest.TestCase):
    def setUp(self):
        """Set up masked mean pooling model instance."""
        self.model = PtnMaskedMeanPooling().f()[0]
        self.model.eval()

        # Each case: (inputs, mask, expected_output)
        self.test_cases = [
            (
                torch.tensor([
                    [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
                    [[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]]
                ]),
                torch.tensor([
                    [1, 1, 0],
                    [1, 0, 0]
                ]),
                torch.tensor([
                    [(1.0 + 3.0)/2, (2.0 + 4.0)/2],
                    [5.0, 5.0]
                ])
            ),
            (
                torch.tensor([
                    [[2.0, 4.0], [6.0, 8.0]]
                ]),
                torch.tensor([
                    [0, 0]
                ]),
                torch.tensor([
                    [0.0, 0.0]
                ])
            ),
            (
                torch.tensor([
                    [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
                ]),
                torch.tensor([
                    [1, 1, 1]
                ]),
                torch.tensor([
                    [(1+2+3)/3, (1+2+3)/3]
                ])
            )
        ]

    def test_masked_mean_pooling(self):
        """Test masked mean pooling against expected results."""
        for i, (inputs, mask, expected) in enumerate(self.test_cases):
            with self.subTest(i=i):
                output = self.model(inputs, mask)
                self.assertTrue(torch.allclose(output, expected, atol=1e-5), f"Mismatch in case {i}: got {output}, expected {expected}")

    def test_invalid_mask_rank(self):
        """Test model raises ValueError for non-2D mask."""
        inputs = torch.randn(2, 3, 4)
        invalid_mask = torch.ones(2, 3, 1)
        with self.assertRaises(ValueError):
            self.model(inputs, invalid_mask)

    def test_invalid_input_rank(self):
        """Test model raises ValueError for non-3D input."""
        inputs = torch.randn(2, 3)
        mask = torch.ones(2, 3)
        with self.assertRaises(ValueError):
            self.model(inputs, mask)

if __name__ == "__main__":
    unittest.main()
