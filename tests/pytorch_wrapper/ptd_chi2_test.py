import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_chi2 import PtdChi2

class TestPtdChi2(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdChi2()

    def test_chi2_distribution_scalar(self):
        """Test instantiation of chi2 distribution."""
        dist = self.node.f("10.0")[0]
        self.assertTrue(isinstance(dist, torch.distributions.studentT.Chi2), "Probability distribution is not Chi2")

    def test_chi2_distribution_1d(self):
        """Test instantiation of chi2 distribution."""
        dist = self.node.f("(10.0,)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.studentT.Chi2), "Probability distribution is not Chi2")

    def test_chi2_distribution_2d(self):
        """Test instantiation of chi2 distribution."""
        dist = self.node.f("(10.0, 11.0)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.studentT.Chi2), "Probability distribution is not Chi2")

   
if __name__ == "__main__":
    unittest.main()
