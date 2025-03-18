import os
import sys
import unittest
import torch
import scipy.stats as scst

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_binomial import PtdBinomial
from pytorch_wrapper.ptd_poisson import PtdPoisson
from pytorch_wrapper.ptdm_pmf import PtdmPmf


class TestPtdfPmf(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        # Binomial
        test_cases = []
        n = 10
        p = 0.5
        for k in range(n+1):
            e = (n, k, p, scst.binom.pmf(k, n, p))
            test_cases.append(e)

        p = 0.01
        for k in range(n+1):
            e = (n, k, p, scst.binom.pmf(k, n, p))
            test_cases.append(e)

        p = 0.99
        for k in range(n+1):
            e = (n, k, p, scst.binom.pmf(k, n, p))
            test_cases.append(e)

        self.test_cases_binomial = test_cases

        # Poisson
        test_cases = []
        lamb = 3  # rate
        n = 10
        for k in range(n+1):
            e = (k, lamb, scst.poisson.pmf(k, lamb))
            test_cases.append(e)
        self.test_cases_poisson = test_cases

    def test_binomial_distribution(self):
        """Test binomial probability mass."""
        for n, k, p, expected_mass in self.test_cases_binomial:
            with self.subTest(n=n, k=k, p=p):
                distribution_node = PtdBinomial().f(str(n), str(p), "")[0]
                prob = PtdmPmf().f(distribution_node, k)[0]

                # Ensure return values are tensors
                self.assertTrue(isinstance(prob, torch.Tensor), "Probability output is not a tensor")
 
                # Convert tensors to floats for comparison
                prob = prob.item()

                # Check if computed values are approximately equal to expected values
                self.assertAlmostEqual(prob, expected_mass, places=5, 
                                       msg=f"Expected probability {expected_mass}, got {prob}")

    def test_poisson_distribution(self):
        """Test poisson probability."""
        for k, lamb, expected_mass in self.test_cases_poisson:
            with self.subTest(k=k):
                distribution_node = PtdPoisson().f(str(lamb))[0]
                prob = PtdmPmf().f(distribution_node, k)[0]

                # Ensure return values are tensors
                self.assertTrue(isinstance(prob, torch.Tensor), "Probability output is not a tensor")
 
                # Convert tensors to floats for comparison
                prob = prob.item()

                # Check if computed values are approximately equal to expected values
                self.assertAlmostEqual(prob, expected_mass, places=5, 
                                       msg=f"Expected probability {expected_mass}, got {prob}")


if __name__ == "__main__":
    unittest.main()
