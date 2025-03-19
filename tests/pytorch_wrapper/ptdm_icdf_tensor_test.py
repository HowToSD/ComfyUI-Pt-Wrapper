import os
import sys
import unittest
import numpy as np
import torch
import scipy.stats as scst
from torch.distributions import Exponential

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptdm_icdf_tensor import PtdmIcdfTensor
from .utils import assert_equal_for_tensor_and_list_or_scalar

class TestPtdfIcdfTensor(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.test_cases = {
            "Exponential": [
                (1.0, 0.9, scst.expon.ppf(0.9, scale=1)),
                (1.0, [0.8, 0.9], scst.expon.ppf(np.array([0.8, 0.9], dtype=np.float32), scale=1))
            ],
        }

    def test_distributions(self):
        """Test CDF."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    if dist_name == "Exponential":
                        distribution = Exponential(torch.tensor(case[0]))
                        k_tensor = torch.tensor(case[1], dtype=torch.float32)
                    else:
                        raise ValueError("Only exponential is supported in this unit test.")

                    cdf = PtdmIcdfTensor().f(distribution, k_tensor)[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(cdf, torch.Tensor), "Output is not a tensor")

                    # Check values are equal
                    assert_equal_for_tensor_and_list_or_scalar(self, cdf, case[-1])
                    

if __name__ == "__main__":
    unittest.main()
