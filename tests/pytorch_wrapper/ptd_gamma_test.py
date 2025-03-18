import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_gamma import PtdGamma

class TestPtdGamma(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdGamma()

    def test_gamma_distribution_scalar(self):
        """Test instantiation of gamma distribution."""
        dist = self.node.f("1.0","1.0")[0]
        self.assertTrue(isinstance(dist, torch.distributions.gamma.Gamma), "Probability distribution is not Gamma")

    def test_gamma_distribution_1d(self):
        """Test instantiation of gamma distribution."""
        dist = self.node.f("(1.0,)", "(1.0,)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.gamma.Gamma), "Probability distribution is not Gamma")

    def test_gamma_distribution_2d(self):
        """Test instantiation of gamma distribution."""
        dist = self.node.f("(1.0,1.0)","(1.0,1.0)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.gamma.Gamma), "Probability distribution is not Gamma")

   
if __name__ == "__main__":
    unittest.main()
