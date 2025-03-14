import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_linear import PtnLinear


class TestPtnLinearModel(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnLinear()

    def test_1(self):
        self.model = self.model_node.f(
            2,  # in
            32,  # out
            True # bias
        )[0]
        x = torch.ones(4, 2)
        out = self.model(x)
        self.assertEqual(out.size(), torch.Size([4, 32]))
        # e contains [0] name and [1] param value, so extract the first element.
        param_names = list(
            map(lambda e: e[0], self.model.named_parameters()))
        self.assertTrue("bias" in param_names)

    def test_2(self):
        self.model = self.model_node.f(
            2,  # in
            1,  # out
            False # bias
        )[0]
        x = torch.ones(4, 2)
        out = self.model(x)
        self.assertEqual(out.size(), torch.Size([4, 1]))
        # e contains [0] name and [1] param value, so extract the first element.
        param_names = list(
            map(lambda e: e[0], self.model.named_parameters()))
        self.assertFalse("bias" in param_names)


if __name__ == "__main__":
    unittest.main()
