import os
import sys
import unittest
import torch


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pto_sgd import PtoSGD


class TestPtoSGD(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtoSGD()

    def test_sgd(self):
        model = torch.nn.Linear(2, 3, bias=False)
        torch.nn.init.constant_(model.weight, 5.0)
        model.weight.grad = torch.full_like(model.weight, 1.0)

        # Hyperparameters
        learning_rate = 0.1
        mu = 0.9
        tau = 0.9
        weight_decay = 0.01
        
        opt = self.node.f(model, learning_rate=learning_rate, 
                          momentum=mu,
                          dampening=tau,
                          weight_decay=weight_decay,
                          nesterov=False)[0]

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
