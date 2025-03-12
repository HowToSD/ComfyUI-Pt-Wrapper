import os
import sys
import unittest
import torch
import torch.optim as optim

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pto_lr_scheduler_reduce_on_plateau import PtoLrSchedulerReduceOnPlateau  # Updated import


class TestPtoLrSchedulerReduceOnPlateau(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtoLrSchedulerReduceOnPlateau()

        # Create a pseudo optimizer with learning rate parameter
        self.model = torch.nn.Linear(2, 2)  # Dummy model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.grace_period = 3
        self.gamma = 0.5

    def test_lr_scheduler_reduce_on_plateau(self):
        """Test if ReduceLROnPlateau correctly updates learning rate."""
        scheduler, = self.node.f(self.optimizer, self.grace_period, self.gamma)

        # Ensure returned object is ReduceLROnPlateau
        self.assertIsInstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau, "Returned object is not a ReduceLROnPlateau scheduler")

        initial_lr = self.optimizer.param_groups[0]['lr']
        # No improvement in loss
        # Grace period map:   0,   1    2    3
        # First epoch is the baseline, so it does not count.
        # In this test case, the number 4 (the 4th) bad loss is greater than grace_period=3,
        # so the loss reduction happens there.
        loss_values = [0.5, 0.5, 0.5, 0.5, 0.5]  
        for loss in loss_values:
            scheduler.step(loss)
        updated_lr = self.optimizer.param_groups[0]['lr']

        # Check if learning rate was updated correctly after grace period
        expected_lr = initial_lr * self.gamma
        self.assertAlmostEqual(updated_lr, expected_lr, places=6, msg=f"Expected LR {expected_lr}, got {updated_lr}")

    def test_lr_scheduler_reduce_on_plateau_no_apply(self):
        """Test if ReduceLROnPlateau correctly does not prematurely update learning rate."""
        scheduler, = self.node.f(self.optimizer, self.grace_period, self.gamma)

        # Ensure returned object is ReduceLROnPlateau
        self.assertIsInstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau, "Returned object is not a ReduceLROnPlateau scheduler")

        initial_lr = self.optimizer.param_groups[0]['lr']
        loss_values = [0.5, 0.5, 0.5, 0.5]  
        for loss in loss_values:
            scheduler.step(loss)
        updated_lr = self.optimizer.param_groups[0]['lr']

        expected_lr = initial_lr
        self.assertAlmostEqual(updated_lr, expected_lr, places=6, msg=f"Expected LR {expected_lr}, got {updated_lr}")


if __name__ == "__main__":
    unittest.main()
