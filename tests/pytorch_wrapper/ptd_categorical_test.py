import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_categorical import PtdCategorical

class TestPtdCategorical(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdCategorical()

    def test_categorical_distribution_probs(self):
        """Test instantiation of categorical distribution."""
        dist = self.node.f("[0.1,0.2,0.7]", "")[0]
        self.assertTrue(isinstance(dist, torch.distributions.categorical.Categorical), "Probability distribution is not Categorical")

    def test_categorical_distribution_logits(self):
        """Test instantiation of categorical distribution."""
        dist = self.node.f("", "[-40, -0.40, 0.36797678529459443]")[0]
        self.assertTrue(isinstance(dist, torch.distributions.categorical.Categorical), "Probability distribution is not Categorical")


if __name__ == "__main__":
    unittest.main()
