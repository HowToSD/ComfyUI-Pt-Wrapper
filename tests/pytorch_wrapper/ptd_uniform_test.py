import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_uniform import PtdUniform

class TestPtdUniform(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdUniform()

    def test_uniform_distribution_scalar(self):
        """Test instantiation of uniform distribution."""
        dist = self.node.f("0.0","1.0")[0]
        self.assertTrue(isinstance(dist, torch.distributions.uniform.Uniform), "Probability distribution is not Uniform")

    def test_uniform_distribution_1d(self):
        """Test instantiation of uniform distribution."""
        dist = self.node.f("(0.0,)", "(1.0,)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.uniform.Uniform), "Probability distribution is not Uniform")

    def test_uniform_distribution_2d(self):
        """Test instantiation of uniform distribution."""
        dist = self.node.f("(155.0,170.0)","(160.0,175.0)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.uniform.Uniform), "Probability distribution is not Uniform")

   
if __name__ == "__main__":
    unittest.main()
