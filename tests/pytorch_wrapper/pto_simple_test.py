import os
import sys
import unittest
import torch


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pto_simple import PtoSimple


class TestPtoSimple(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtoSimple()

    def test_1(self):
        model = torch.nn.Linear(2, 3, bias=False)
        torch.nn.init.constant_(model.weight, 5.0)
        model.weight.grad = torch.full_like(model.weight, 1.0)
        opt = self.node.f(model, 0.1)[0]
        opt.step()  # 5.0 - 0.1 * 1.0 = 4.9
        expected = 4.9
        result = model.weight.data[0, 0]
        self.assertAlmostEqual(
            result.item(), 
            expected,
            places=6, 
            msg=f"Expected {expected}, got {result}")


if __name__ == "__main__":
    unittest.main()
