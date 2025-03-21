import os
import sys
import unittest
import torch
from torch import Tensor
from typing import List, Tuple

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_cross_entropy_loss import PtnCrossEntropyLoss 


class TestPtCrossEntropyLoss(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test instance by initializing the loss functions and test cases.
        """
        self.node = PtnCrossEntropyLoss()
        self.loss_fun_none = self.node.f("none")[0]
        self.loss_fun_mean = self.node.f("mean")[0]
        self.loss_fun_sum = self.node.f("sum")[0]

        self.test_cases: List[Tuple[Tensor, Tensor]] = [
            (
                torch.tensor([
                    [-2.0, -1.0, 0.0, 1.0, 2.0],
                    [-40.0, -20.0, 0.0, 10.0, 20.0]
                ], dtype=torch.float32),
                torch.tensor([0, 2], dtype=torch.int64)
            )
        ]

    def unreduced_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the unreduced cross entropy loss.

        Args:
            logits (Tensor): Raw, unnormalized class scores (logits).
            targets (Tensor): Ground truth class indices.

        Returns:
            Tensor: Unreduced element-wise cross entropy loss.
        """
        return torch.nn.functional.cross_entropy(logits, targets, reduction='none')

    def test_none(self) -> None:
        """
        Test cross entropy loss with reduction='none'.
        """
        for logits, targets in self.test_cases:
            with self.subTest(logits=logits):
                expected = self.unreduced_loss(logits, targets)
                actual = self.loss_fun_none(logits, targets)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-4),
                                f"Expected {expected}, got {actual}")

    def test_mean(self) -> None:
        """
        Test cross entropy loss with reduction='mean'.
        """
        for logits, targets in self.test_cases:
            with self.subTest(logits=logits):
                loss = self.unreduced_loss(logits, targets)
                expected = loss.mean(dim=0)
                actual = self.loss_fun_mean(logits, targets)
                self.assertTrue(torch.equal(actual, expected),
                                f"Expected {expected}, got {actual}")

    def test_sum(self) -> None:
        """
        Test cross entropy loss with reduction='sum'.
        """
        for logits, targets in self.test_cases:
            with self.subTest(logits=logits):
                loss = self.unreduced_loss(logits, targets)
                expected = loss.sum(dim=0)
                actual = self.loss_fun_sum(logits, targets)
                self.assertTrue(torch.equal(actual, expected),
                                f"Expected {expected}, got {actual}")


if __name__ == "__main__":
    unittest.main()
