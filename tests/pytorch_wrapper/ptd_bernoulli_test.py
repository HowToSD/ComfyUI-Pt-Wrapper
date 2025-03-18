import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_bernoulli import PtdBernoulli

class TestPtdBernoulli(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdBernoulli()

    def test_bernoulli_distribution_probs1(self):
        """Test instantiation of bernoulli distribution."""
        dist = self.node.f("0", "")[0]  # p=0.5
        self.assertTrue(isinstance(dist, torch.distributions.bernoulli.Bernoulli), "Probability distribution is not Bernoulli")

    def test_bernoulli_distribution_probs2(self):
        """Test instantiation of bernoulli distribution."""
        dist = self.node.f("0.5", "")[0]  # p=0.5
        self.assertTrue(isinstance(dist, torch.distributions.bernoulli.Bernoulli), "Probability distribution is not Bernoulli")

    def test_bernoulli_distribution_probs3(self):
        """Test instantiation of bernoulli distribution."""
        dist = self.node.f("1", "")[0]  # p=0.5
        self.assertTrue(isinstance(dist, torch.distributions.bernoulli.Bernoulli), "Probability distribution is not Bernoulli")

    def test_bernoulli_distribution_logits1(self):
        """Test instantiation of bernoulli distribution."""
        dist = self.node.f("", "-40")[0]
        self.assertTrue(isinstance(dist, torch.distributions.bernoulli.Bernoulli), "Probability distribution is not Bernoulli")

    def test_bernoulli_distribution_logits2(self):
        """Test instantiation of bernoulli distribution."""
        dist = self.node.f("", "0")[0]
        self.assertTrue(isinstance(dist, torch.distributions.bernoulli.Bernoulli), "Probability distribution is not Bernoulli")

    def test_bernoulli_distribution_logits3(self):
        """Test instantiation of bernoulli distribution."""
        dist = self.node.f("", "40")[0]
        self.assertTrue(isinstance(dist, torch.distributions.bernoulli.Bernoulli), "Probability distribution is not Bernoulli")


   
if __name__ == "__main__":
    unittest.main()
