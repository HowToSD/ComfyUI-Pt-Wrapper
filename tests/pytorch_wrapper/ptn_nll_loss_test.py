import os
import sys
import unittest
import torch
from torch import Tensor
from typing import List, Tuple

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_nll_loss import PtnNLLLoss


class TestPtnNLLLoss(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test instance by initializing the loss functions and test cases.
        """
        self.node = PtnNLLLoss()
        self.loss_fun_none = self.node.f("none")[0]
        self.loss_fun_mean = self.node.f("mean")[0]
        self.loss_fun_sum = self.node.f("sum")[0]

        logits = torch.tensor([
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-40.0, -20.0, 0.0, 10.0, 20.0]
        ], dtype=torch.float32)

        log_probs = torch.nn.functional.log_softmax(logits, dim=1)

        self.test_cases: List[Tuple[Tensor, Tensor]] = [
            (
                log_probs,
                torch.tensor([0, 2], dtype=torch.int64)
            )
        ]

    def unreduced_loss(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the unreduced negative log-likelihood loss.

        Args:
            log_probs (Tensor): Log-probabilities from log_softmax.
            targets (Tensor): Ground truth class indices.

        Returns:
            Tensor: Unreduced element-wise NLL loss.
        """
        return torch.nn.functional.nll_loss(log_probs, targets, reduction='none')

    def test_none(self) -> None:
        """
        Test NLL loss with reduction='none'.
        """
        for log_probs, targets in self.test_cases:
            with self.subTest(log_probs=log_probs):
                expected = self.unreduced_loss(log_probs, targets)
                actual = self.loss_fun_none(log_probs, targets)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-4),
                                f"Expected {expected}, got {actual}")

    def test_mean(self) -> None:
        """
        Test NLL loss with reduction='mean'.
        """
        for log_probs, targets in self.test_cases:
            with self.subTest(log_probs=log_probs):
                loss = self.unreduced_loss(log_probs, targets)
                expected = loss.mean(dim=0)
                actual = self.loss_fun_mean(log_probs, targets)
                self.assertTrue(torch.equal(actual, expected),
                                f"Expected {expected}, got {actual}")

    def test_sum(self) -> None:
        """
        Test NLL loss with reduction='sum'.
        """
        for log_probs, targets in self.test_cases:
            with self.subTest(log_probs=log_probs):
                loss = self.unreduced_loss(log_probs, targets)
                expected = loss.sum(dim=0)
                actual = self.loss_fun_sum(log_probs, targets)
                self.assertTrue(torch.equal(actual, expected),
                                f"Expected {expected}, got {actual}")


if __name__ == "__main__":
    unittest.main()
