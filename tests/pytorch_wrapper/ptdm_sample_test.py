import os
import sys
import unittest
import torch
import scipy.stats as scst
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_binomial import PtdBinomial
from pytorch_wrapper.ptdm_sample import PtdmSample

class TestPtdfSample(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""

    def test_scalar_binom(self):
        """Test binom scalar."""
        distribution_node = PtdBinomial().f(str(10), str(0.5), "")[0]
        sample = PtdmSample().f(distribution_node, str(5))[0]

        self.assertEqual(sample.size(), (5,))

    def test_1d_binom(self):
        """Test binom 1d."""
        distribution_node = PtdBinomial().f(str(10), str(0.5), "")[0]
        sample = PtdmSample().f(distribution_node, "(5,)")[0]

        self.assertEqual(sample.size(), (5,))

    def test_2d_binom(self):
        """Test binom 2d."""
        distribution_node = PtdBinomial().f(str(10), str(0.5), "")[0]
        sample = PtdmSample().f(distribution_node, ("(2,3)"))[0]

        self.assertEqual(sample.size(), (2,3))


if __name__ == "__main__":
    unittest.main()
