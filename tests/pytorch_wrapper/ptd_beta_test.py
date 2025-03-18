import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_beta import PtdBeta

class TestPtdBeta(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdBeta()

    def test_beta_distribution_scalar(self):
        """Test instantiation of beta distribution."""
        dist = self.node.f("1.0","9.0")[0]
        self.assertTrue(isinstance(dist, torch.distributions.beta.Beta), "Probability distribution is not Beta")

    def test_beta_distribution_1d(self):
        """Test instantiation of beta distribution."""
        dist = self.node.f("(1.0,)", "(9.0,)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.beta.Beta), "Probability distribution is not Beta")

    def test_beta_distribution_2d(self):
        """Test instantiation of beta distribution."""
        dist = self.node.f("(1.0,3.0)","(9.0,10.0)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.beta.Beta), "Probability distribution is not Beta")

   
if __name__ == "__main__":
    unittest.main()
