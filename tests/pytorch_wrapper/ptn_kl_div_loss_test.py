import os
import sys
import unittest
import torch
from torch import Tensor
from typing import List, Tuple

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_kl_div_loss import PtnKLDivLoss 


class TestPtnKLDivLoss(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test instance by initializing the loss functions and test cases.
        """
        self.node = PtnKLDivLoss()
        self.loss_fun_none = self.node.f("none", log_target=False)[0]
        self.loss_fun_batchmean = self.node.f("batchmean", log_target=False)[0]
        self.loss_fun_mean = self.node.f("mean", log_target=False)[0]
        self.loss_fun_sum = self.node.f("sum", log_target=False)[0]

        self.loss_fun_none_log = self.node.f("none", log_target=True)[0]
        self.loss_fun_batchmean_log = self.node.f("batchmean", log_target=True)[0]
        self.loss_fun_mean_log = self.node.f("mean", log_target=True)[0]
        self.loss_fun_sum_log = self.node.f("sum", log_target=True)[0]

        # log_probs: log_softmax of logits
        self.test_cases: List[Tuple[Tensor, Tensor]] = [
            (
                # input (log-probabilities)
                torch.log_softmax(
                    torch.tensor([
                        [-2.0, -1.0, 0.0, 1.0, 2.0],
                        [-40.0, -20.0, 0.0, 10.0, 20.0]
                    ], dtype=torch.float32), dim=-1
                ),
                # target (probabilities for log_target=False)
                torch.tensor([
                    [0.04, 0.06, 0.2, 0.3, 0.4],
                    [0.4, 0.3, 0.2, 0.06, 0.04]
                ], dtype=torch.float32)
            )
        ]

        self.log_target_cases: List[Tuple[Tensor, Tensor]] = [
            (
                # input
                torch.log_softmax(
                    torch.tensor([
                        [-2.0, -1.0, 0.0, 1.0, 2.0],
                        [-40.0, -20.0, 0.0, 10.0, 20.0]
                    ], dtype=torch.float32), dim=-1
                ),
                # target (log-probabilities)
                torch.log_softmax(
                    torch.tensor([
                        [0.04, 0.06, 0.2, 0.3, 0.4],
                        [0.4, 0.3, 0.2, 0.06, 0.04]
                    ], dtype=torch.float32), dim=-1
                )
            )
        ]

    def unreduced_loss(self, log_probs: Tensor, probs: Tensor) -> Tensor:
        return torch.nn.functional.kl_div(log_probs, probs, reduction='none', log_target=False)

    def batchmean_loss(self, log_probs: Tensor, probs: Tensor) -> Tensor:
        return torch.nn.functional.kl_div(log_probs, probs, reduction='batchmean', log_target=False)

    def unreduced_loss_log_target(self, log_probs: Tensor, log_targets: Tensor) -> Tensor:
        return torch.nn.functional.kl_div(log_probs, log_targets, reduction='none', log_target=True)

    def batchmean_loss_log_target(self, log_probs: Tensor, log_targets: Tensor) -> Tensor:
        return torch.nn.functional.kl_div(log_probs, log_targets, reduction='batchmean', log_target=True)

    def test_none(self) -> None:
        for log_probs, probs in self.test_cases:
            expected = self.unreduced_loss(log_probs, probs)
            actual = self.loss_fun_none(log_probs, probs)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-4))

    def test_mean(self) -> None:
        for log_probs, probs in self.test_cases:
            expected = self.unreduced_loss(log_probs, probs).mean()
            actual = self.loss_fun_mean(log_probs, probs)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_batchmean(self) -> None:
        for log_probs, probs in self.test_cases:
            expected = self.batchmean_loss(log_probs, probs)
            actual = self.loss_fun_batchmean(log_probs, probs)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_sum(self) -> None:
        for log_probs, probs in self.test_cases:
            expected = self.unreduced_loss(log_probs, probs).sum()
            actual = self.loss_fun_sum(log_probs, probs)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_none_log_target(self) -> None:
        for log_probs, log_targets in self.log_target_cases:
            expected = self.unreduced_loss_log_target(log_probs, log_targets)
            actual = self.loss_fun_none_log(log_probs, log_targets)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-4))

    def test_mean_log_target(self) -> None:
        for log_probs, log_targets in self.log_target_cases:
            expected = self.unreduced_loss_log_target(log_probs, log_targets).mean()
            actual = self.loss_fun_mean_log(log_probs, log_targets)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_batchmean_log_target(self) -> None:
        for log_probs, log_targets in self.log_target_cases:
            expected = self.batchmean_loss_log_target(log_probs, log_targets)
            actual = self.loss_fun_batchmean_log(log_probs, log_targets)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_sum_log_target(self) -> None:
        for log_probs, log_targets in self.log_target_cases:
            expected = self.unreduced_loss_log_target(log_probs, log_targets).sum()
            actual = self.loss_fun_sum_log(log_probs, log_targets)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
