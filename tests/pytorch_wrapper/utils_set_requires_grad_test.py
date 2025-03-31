import os
import sys
import unittest
import torch.nn as nn


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.utils import set_requires_grad


class TestSetRequiresGrad(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(2, 3, bias=True)

    def test_1(self):
        # Check that gradients are initially enabled
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

        # Turn off gradients
        set_requires_grad(self.model, False)
        for param in self.model.parameters():
            self.assertFalse(param.requires_grad)

        # Turn on gradients
        set_requires_grad(self.model, True)
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main()

