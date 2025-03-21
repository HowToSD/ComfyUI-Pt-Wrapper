import os
import sys
import unittest
import torch
from torch import Tensor
from typing import List, Tuple

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_bce_loss import PtnBCELoss


class TestPtnBCELoss(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test instance by initializing the loss functions and test cases.
        """
        self.node = PtnBCELoss()
        self.loss_fun_none = self.node.f("none")[0]
        self.loss_fun_mean = self.node.f("mean")[0]
        self.loss_fun_sum = self.node.f("sum")[0]

        # Test cases: (predicted probabilities y_hat, ground truth y)
        self.test_cases: List[Tuple[Tensor, Tensor]] = [
            (
                torch.tensor([0.1, 0.9], dtype=torch.float32),
                torch.tensor([1, 0], dtype=torch.float32),
            )
        ]

    def unreduced_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Compute the unreduced binary cross-entropy loss.

        Args:
            y_hat (Tensor): Predicted probabilities.
            y (Tensor): Ground truth binary labels.

        Returns:
            Tensor: Unreduced element-wise BCE loss.
        """
        return -(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))

    def test_none(self) -> None:
        """
        Test BCE loss with reduction='none'.
        """
        for y_hat, y in self.test_cases:
            with self.subTest(y_hat=y_hat):
                expected = self.unreduced_loss(y_hat, y)
                actual = self.loss_fun_none(y_hat, y)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-4),
                                f"Expected {expected}, got {actual}")

    def test_mean(self) -> None:
        """
        Test BCE loss with reduction='mean'.
        """
        for y_hat, y in self.test_cases:
            with self.subTest(y_hat=y_hat):
                loss = self.unreduced_loss(y_hat, y)
                expected = loss.mean(dim=0)
                actual = self.loss_fun_mean(y_hat, y)
                self.assertTrue(torch.equal(actual, expected),
                                f"Expected {expected}, got {actual}")

    def test_sum(self) -> None:
        """
        Test BCE loss with reduction='sum'.
        """
        for y_hat, y in self.test_cases:
            with self.subTest(y_hat=y_hat):
                loss = self.unreduced_loss(y_hat, y)
                expected = loss.sum(dim=0)
                actual = self.loss_fun_sum(y_hat, y)
                self.assertTrue(torch.equal(actual, expected),
                                f"Expected {expected}, got {actual}")


if __name__ == "__main__":
    unittest.main()
