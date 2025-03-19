import os
import sys
import unittest
import torch
import scipy.stats as scst
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_binomial import BinomialEx
from pytorch_wrapper.ptd_binomial import PtdBinomial
from pytorch_wrapper.ptdm_cdf_tensor import PtdmCdfTensor
from .utils import assert_equal_for_tensor_and_list_or_scalar

class TestPtdfCdfBinomial(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.test_cases = {
            "Binomial": [
                (10.0, 3.0, 0.4, scst.binom.cdf(3.0, 10, 0.4)),
                (10.0, [3.0, 4.0], 0.4, scst.binom.cdf([3.0, 4.0], 10, 0.4))  # 1d array for x
            ]
        }

    def test_custom_class_1(self):
        d = BinomialEx(total_count=10, probs=0.4)
        actual = d.cdf(torch.tensor(3.0)).numpy()
        expected = torch.tensor(0.3822806015999999).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_2(self):
        d = BinomialEx(total_count=10, probs=0.4)
        actual = d.cdf(torch.tensor(1.0)).numpy()
        expected = torch.tensor(0.04635740159999998).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_3(self):
        d = BinomialEx(total_count=10, probs=0.4)
        actual = d.cdf(torch.tensor([1.0, 3.0])).numpy()
        expected = torch.tensor([0.04635740159999998, 0.3822806015999999]).numpy()
        self.assertTrue(np.allclose(actual, expected, atol=1e-5), 
                                msg=f"Expected {expected}, got {actual}")

    def test_distributions(self):
        """Test probability mass and density functions for the target distribution."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    distribution = PtdBinomial().f(str(case[0]), str(case[2]), "")[0]  # n, p
                    cdf = PtdmCdfTensor().f(
                        distribution,
                        torch.tensor(case[1]))[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(cdf, torch.Tensor), "Probability output is not a tensor")

                    # Check values are equal
                    assert_equal_for_tensor_and_list_or_scalar(self, cdf, case[-1])


if __name__ == "__main__":
    unittest.main()
