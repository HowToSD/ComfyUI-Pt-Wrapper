import os
import sys
import unittest
import torch
import torch.optim as optim

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pto_lr_scheduler_step import PtoLrSchedulerStep  # Updated import


class TestPtoLrSchedulerStep(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtoLrSchedulerStep()

        # Create a pseudo optimizer with learning rate parameter
        # For every 5 steps, if gamma is 0.1, then lr *= 0.5
        # so 0.1 -> 0.05 -> 0.025
        self.model = torch.nn.Linear(2, 2)  # Dummy model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.step_size = 5
        self.gamma = 0.1

    def test_lr_scheduler_step(self):
        """Test if learning rate scheduler correctly updates learning rate."""
        scheduler, = self.node.f(self.optimizer, self.step_size, self.gamma)

        # Ensure returned object is StepLR
        self.assertIsInstance(scheduler, optim.lr_scheduler.StepLR, "Returned object is not a StepLR scheduler")

        initial_lr = self.optimizer.param_groups[0]['lr']
        for _ in range(self.step_size):
            scheduler.step()
        updated_lr = self.optimizer.param_groups[0]['lr']

        # Check if learning rate was updated correctly
        expected_lr = initial_lr * self.gamma
        self.assertAlmostEqual(updated_lr, expected_lr, places=6, msg=f"Expected LR {expected_lr}, got {updated_lr}")

if __name__ == "__main__":
    unittest.main()
