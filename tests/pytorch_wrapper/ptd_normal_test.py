import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_normal import PtdNormal

class TestPtdNormal(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdNormal()

    def test_normal_distribution_scalar(self):
        """Test instantiation of normal distribution."""
        dist = self.node.f("0.0","1.0")[0]
        self.assertTrue(isinstance(dist, torch.distributions.normal.Normal), "Probability distribution is not Normal")

    def test_normal_distribution_1d(self):
        """Test instantiation of normal distribution."""
        dist = self.node.f("(0.0,)", "(1.0,)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.normal.Normal), "Probability distribution is not Normal")

    def test_normal_distribution_2d(self):
        """Test instantiation of normal distribution."""
        dist = self.node.f("(155.0,170.0)","(8.0,10.0)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.normal.Normal), "Probability distribution is not Normal")

   
if __name__ == "__main__":
    unittest.main()
