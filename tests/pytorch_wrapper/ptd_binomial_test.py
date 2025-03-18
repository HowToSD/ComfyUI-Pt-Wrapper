import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_binomial import PtdBinomial

class TestPtdBinomial(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdBinomial()

    def test_binomial_distribution_probs1(self):
        """Test instantiation of binomial distribution."""
        dist = self.node.f("10", "0", "")[0]  # n = 10, p=0
        self.assertTrue(isinstance(dist, torch.distributions.binomial.Binomial), "Probability distribution is not Binomial")

    def test_binomial_distribution_probs2(self):
        """Test instantiation of binomial distribution."""
        dist = self.node.f("10", "0.0", "")[0]  # n = 10, p=0
        self.assertTrue(isinstance(dist, torch.distributions.binomial.Binomial), "Probability distribution is not Binomial")

    def test_binomial_distribution_probs3(self):
        """Test instantiation of binomial distribution."""
        dist = self.node.f("10", "0.5", "")[0]  # n = 10, p=0.5
        self.assertTrue(isinstance(dist, torch.distributions.binomial.Binomial), "Probability distribution is not Binomial")

    def test_binomial_distribution_probs4(self):
        """Test instantiation of binomial distribution."""
        dist = self.node.f("10", "1", "")[0]  # n = 10, p=1
        self.assertTrue(isinstance(dist, torch.distributions.binomial.Binomial), "Probability distribution is not Binomial")

    def test_binomial_distribution_probs5(self):
        """Test instantiation of binomial distribution."""
        dist = self.node.f("10", "1.0", "")[0]  # n = 10, p=1
        self.assertTrue(isinstance(dist, torch.distributions.binomial.Binomial), "Probability distribution is not Binomial")

    def test_binomial_distribution_logits1(self):
        """Test instantiation of binomial distribution."""
        dist = self.node.f("10", "", "-40")[0]
        self.assertTrue(isinstance(dist, torch.distributions.binomial.Binomial), "Probability distribution is not Binomial")

    def test_binomial_distribution_logits2(self):
        """Test instantiation of binomial distribution."""
        dist = self.node.f("10", "", "0")[0]
        self.assertTrue(isinstance(dist, torch.distributions.binomial.Binomial), "Probability distribution is not Binomial")

    def test_binomial_distribution_logits3(self):
        """Test instantiation of binomial distribution."""
        dist = self.node.f("10", "", "40")[0]
        self.assertTrue(isinstance(dist, torch.distributions.binomial.Binomial), "Probability distribution is not Binomial")


if __name__ == "__main__":
    unittest.main()
