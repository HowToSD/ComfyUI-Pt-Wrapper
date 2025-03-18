import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_exponential import PtdExponential

class TestPtdExponential(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdExponential()

    def test_exponential_distribution_scalar(self):
        """Test instantiation of exponential distribution."""
        dist = self.node.f("3")[0]
        self.assertTrue(isinstance(dist, torch.distributions.exponential.Exponential), "Probability distribution is not Exponential")

    def test_exponential_distribution_1d(self):
        """Test instantiation of exponential distribution."""
        dist = self.node.f("(3,)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.exponential.Exponential), "Probability distribution is not Exponential")

    def test_exponential_distribution_2d(self):
        """Test instantiation of exponential distribution."""
        dist = self.node.f("(2,3)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.exponential.Exponential), "Probability distribution is not Exponential")

   
if __name__ == "__main__":
    unittest.main()
