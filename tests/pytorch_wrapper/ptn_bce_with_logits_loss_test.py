import os
import sys
import unittest
import torch
from torch import Tensor
from typing import List, Tuple

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_bce_with_logits_loss import PtnBCEWithLogitsLoss


class TestPtnBCEWithLogits(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test instance by initializing the loss functions and test cases.
        """
        self.node = PtnBCEWithLogitsLoss()
        self.loss_fun_none = self.node.f("none")[0]
        self.loss_fun_mean = self.node.f("mean")[0]
        self.loss_fun_sum = self.node.f("sum")[0]

        # Probabilities: 0.1, 0.8 → Logits: log(0.1/0.9), log(0.8/0.2)
        logits = torch.log(torch.tensor([0.1, 0.8]) / torch.tensor([0.9, 0.2]))
        self.test_cases: List[Tuple[Tensor, Tensor]] = [
            (
                logits.to(dtype=torch.float32),
                torch.tensor([1, 0], dtype=torch.float32),
            )
        ]

    def unreduced_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Compute the unreduced BCEWithLogits loss.

        Args:
            logits (Tensor): Raw logits.
            y (Tensor): Ground truth binary labels.

        Returns:
            Tensor: Unreduced element-wise BCEWithLogits loss.
        """
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')

    def test_none(self) -> None:
        """
        Test BCEWithLogits loss with reduction='none'.
        """
        for logits, y in self.test_cases:
            with self.subTest(logits=logits):
                expected = self.unreduced_loss(logits, y)
                actual = self.loss_fun_none(logits, y)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-4),
                                f"Expected {expected}, got {actual}")

    def test_mean(self) -> None:
        """
        Test BCEWithLogits loss with reduction='mean'.
        """
        for logits, y in self.test_cases:
            with self.subTest(logits=logits):
                loss = self.unreduced_loss(logits, y)
                expected = loss.mean(dim=0)
                actual = self.loss_fun_mean(logits, y)
                self.assertTrue(torch.equal(actual, expected),
                                f"Expected {expected}, got {actual}")

    def test_sum(self) -> None:
        """
        Test BCEWithLogits loss with reduction='sum'.
        """
        for logits, y in self.test_cases:
            with self.subTest(logits=logits):
                loss = self.unreduced_loss(logits, y)
                expected = loss.sum(dim=0)
                actual = self.loss_fun_sum(logits, y)
                self.assertTrue(torch.equal(actual, expected),
                                f"Expected {expected}, got {actual}")


if __name__ == "__main__":
    unittest.main()
