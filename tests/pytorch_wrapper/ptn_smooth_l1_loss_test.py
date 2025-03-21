import os
import sys
import unittest
import torch
from torch import Tensor
from typing import List, Tuple

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_smooth_l1_loss import PtnSmoothL1Loss


class TestPtnSmoothL1Loss(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test instance by initializing the loss functions and test cases.
        """
        self.node = PtnSmoothL1Loss()

        self.loss_fun_none_1 = self.node.f("none", 0.01)[0]
        self.loss_fun_none_2 = self.node.f("none", 1)[0]
        self.loss_fun_none_3 = self.node.f("none", 100)[0]

        self.loss_fun_mean_1 = self.node.f("mean", 0.01)[0]
        self.loss_fun_mean_2 = self.node.f("mean", 1)[0]
        self.loss_fun_mean_3 = self.node.f("mean", 100)[0]

        self.loss_fun_sum_1 = self.node.f("sum", 0.01)[0]
        self.loss_fun_sum_2 = self.node.f("sum", 1)[0]
        self.loss_fun_sum_3 = self.node.f("sum", 100)[0]

        # Test cases: (predicted values y_hat, ground truth y)
        self.test_cases: List[Tuple[Tensor, Tensor]] = [
            (
                torch.tensor([0.1, -1.0, 10, -100, 1000], dtype=torch.float32),
                torch.tensor([0.09, -1.1, 9, -101, 999], dtype=torch.float32),
            )
        ]

    def unreduced_loss(self, y_hat: Tensor, y: Tensor, beta: float) -> Tensor:
        return torch.nn.functional.smooth_l1_loss(y_hat, y, beta=beta, reduction="none")

    def test_none_beta_001(self) -> None:
        for y_hat, y in self.test_cases:
            expected = self.unreduced_loss(y_hat, y, beta=0.01)
            actual = self.loss_fun_none_1(y_hat, y)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6),
                            f"Expected {expected}, got {actual}")

    def test_none_beta_1(self) -> None:
        for y_hat, y in self.test_cases:
            expected = self.unreduced_loss(y_hat, y, beta=1)
            actual = self.loss_fun_none_2(y_hat, y)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6),
                            f"Expected {expected}, got {actual}")

    def test_none_beta_100(self) -> None:
        for y_hat, y in self.test_cases:
            expected = self.unreduced_loss(y_hat, y, beta=100)
            actual = self.loss_fun_none_3(y_hat, y)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6),
                            f"Expected {expected}, got {actual}")

    def test_mean_beta_001(self) -> None:
        for y_hat, y in self.test_cases:
            loss = self.unreduced_loss(y_hat, y, beta=0.01)
            expected = loss.mean(dim=0)
            actual = self.loss_fun_mean_1(y_hat, y)
            self.assertTrue(torch.equal(actual, expected),
                            f"Expected {expected}, got {actual}")

    def test_mean_beta_1(self) -> None:
        for y_hat, y in self.test_cases:
            loss = self.unreduced_loss(y_hat, y, beta=1)
            expected = loss.mean(dim=0)
            actual = self.loss_fun_mean_2(y_hat, y)
            self.assertTrue(torch.equal(actual, expected),
                            f"Expected {expected}, got {actual}")

    def test_mean_beta_100(self) -> None:
        for y_hat, y in self.test_cases:
            loss = self.unreduced_loss(y_hat, y, beta=100)
            expected = loss.mean(dim=0)
            actual = self.loss_fun_mean_3(y_hat, y)
            self.assertTrue(torch.equal(actual, expected),
                            f"Expected {expected}, got {actual}")

    def test_sum_beta_001(self) -> None:
        for y_hat, y in self.test_cases:
            loss = self.unreduced_loss(y_hat, y, beta=0.01)
            expected = loss.sum(dim=0)
            actual = self.loss_fun_sum_1(y_hat, y)
            self.assertTrue(torch.equal(actual, expected),
                            f"Expected {expected}, got {actual}")

    def test_sum_beta_1(self) -> None:
        for y_hat, y in self.test_cases:
            loss = self.unreduced_loss(y_hat, y, beta=1)
            expected = loss.sum(dim=0)
            actual = self.loss_fun_sum_2(y_hat, y)
            self.assertTrue(torch.equal(actual, expected),
                            f"Expected {expected}, got {actual}")

    def test_sum_beta_100(self) -> None:
        for y_hat, y in self.test_cases:
            loss = self.unreduced_loss(y_hat, y, beta=100)
            expected = loss.sum(dim=0)
            actual = self.loss_fun_sum_3(y_hat, y)
            self.assertTrue(torch.equal(actual, expected),
                            f"Expected {expected}, got {actual}")


if __name__ == "__main__":
    unittest.main()
