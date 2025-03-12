import os
import sys
import unittest
import torch
import torch.optim as optim
import math

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pto_lr_scheduler_cosine_annealing import PtoLrSchedulerCosineAnnealing


class TestPtoLrSchedulerCosineAnnealing(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtoLrSchedulerCosineAnnealing()

        # Create a pseudo optimizer with learning rate parameter
        self.model = torch.nn.Linear(2, 2)  # Dummy model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.num_epochs = 10
        self.minimum_lr = 0.01

    def test_lr_scheduler_cosine_annealing(self):
        """Test if CosineAnnealingLR correctly updates learning rate."""
        scheduler, = self.node.f(self.optimizer, self.num_epochs, self.minimum_lr)

        # Ensure returned object is CosineAnnealingLR
        self.assertIsInstance(scheduler, optim.lr_scheduler.CosineAnnealingLR, "Returned object is not a CosineAnnealingLR scheduler")

        initial_lr = self.optimizer.param_groups[0]['lr']
        for t in range(1, self.num_epochs + 1):
            scheduler.step()
            updated_lr = self.optimizer.param_groups[0]['lr']
            expected_lr = self.minimum_lr + 0.5 * (initial_lr - self.minimum_lr) * (1 + math.cos(math.pi * t / self.num_epochs))
            self.assertAlmostEqual(updated_lr, expected_lr, places=6, msg=f"Expected LR {expected_lr}, got {updated_lr} at step {t}")

if __name__ == "__main__":
    unittest.main()
