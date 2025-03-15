import os
import sys
import unittest
import torch


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pto_adamw import PtoAdamW


class TestPtoAdamW(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtoAdamW()

    def test_adamw(self):
        model = torch.nn.Linear(2, 3, bias=False)
        torch.nn.init.constant_(model.weight, 5.0)
        model.weight.grad = torch.full_like(model.weight, 1.0)

        # AdamW hyperparameters
        learning_rate = 0.1
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01
        eps = 1e-8  # Default AdamW epsilon

        opt = self.node.f(model, learning_rate=learning_rate, beta1=beta1, beta2=beta2, 
                          weight_decay=weight_decay, amsgrad=False)[0]

        expected_weight = torch.tensor(
                [
                    [4.8950, 4.8950],
                    [4.8950, 4.8950],
                    [4.8950, 4.8950]
                ]
            )

        opt.step()
        opt.zero_grad()

        result = model.weight.data

        self.assertTrue(
            torch.allclose(result, expected_weight, atol=1e-6),
            msg=f"Expected {expected_weight}, got {result}"
        )


if __name__ == "__main__":
    unittest.main()
