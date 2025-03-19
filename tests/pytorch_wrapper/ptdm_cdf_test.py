import os
import sys
import unittest
import torch
import scipy.stats as scst
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptdm_cdf import PtdmCdf
from torch.distributions import (
    Categorical, Poisson, NegativeBinomial, Multinomial, 
    Bernoulli, Beta, Gamma, Normal, Laplace, Exponential, LogNormal, 
    Pareto, Weibull, Dirichlet, VonMises, StudentT, Uniform, Chi2
)


class TestPtdfCdf(unittest.TestCase):
    def setUp(self):
        """Set up test instance.
        
        Following distributions are tested in separate unit test files:
        * Bernoulli
        * Binomial
        * Poisson
        * StudentT
        * Beta
        """
        self.test_cases = {
            "Uniform": [(0.0, 1.0, 0.5, scst.uniform.cdf(0.5, loc=0, scale=1))],
            "Normal": [(0.0, 1.0, -1.5, scst.norm.cdf(-1.5, 0, 1))],
            "Chi2": [(1.0, 2.0, scst.chi2.cdf(2.0, 1))],  # df=1, value=2.0
            "Exponential": [(1.0, 3.0, scst.expon.cdf(3.0, scale=1))],
            "Gamma": [(2.0, 1.0, 3.0, scst.gamma.cdf(3.0, 2, scale=1))],
            # "Dirichlet": [(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.2, 0.5, 0.3]), scst.dirichlet.cdf([0.2, 0.5, 0.3], [1.0, 1.0, 1.0]))],
            # "VonMises": [(0.0, 1.0, 1.0, scst.vonmises.cdf(1.0, kappa=1.0, loc=0))],
        }

    def test_distributions(self):
        """Test probability mass and density functions for all PyTorch distributions."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    # Instantiate PyTorch distribution directly if not in wrapper
                    if dist_name == "Categorical":
                        distribution = Categorical(case[0])
                        k_tensor = torch.tensor(case[1], dtype=torch.int64)
                    elif dist_name == "NegativeBinomial":
                        distribution = NegativeBinomial(torch.tensor(case[0]), torch.tensor(case[2]))
                        k_tensor = torch.tensor(case[1], dtype=torch.int64)
                    elif dist_name == "Gamma":
                        distribution = Gamma(torch.tensor(case[0]), torch.tensor(case[1]))
                        k_tensor = torch.tensor(case[2], dtype=torch.float32)
                    elif dist_name == "Normal":
                        distribution = Normal(torch.tensor(case[0]), torch.tensor(case[1]))
                        k_tensor = torch.tensor(case[2], dtype=torch.float32)
                    elif dist_name == "Laplace":
                        distribution = Laplace(torch.tensor(case[0]), torch.tensor(case[1]))
                        k_tensor = torch.tensor(case[2], dtype=torch.float32)
                    elif dist_name == "Exponential":
                        distribution = Exponential(torch.tensor(case[0]))
                        k_tensor = torch.tensor(case[1], dtype=torch.float32)
                    elif dist_name == "LogNormal":
                        distribution = LogNormal(torch.tensor(case[0]), torch.tensor(case[1]))
                        k_tensor = torch.tensor(case[2], dtype=torch.float32)
                    elif dist_name == "Pareto":
                        distribution = Pareto(torch.tensor(case[0]), torch.tensor(case[1]))
                        k_tensor = torch.tensor(case[2], dtype=torch.float32)
                    elif dist_name == "Weibull":
                        distribution = Weibull(torch.tensor(1.0), torch.tensor(case[0]))
                        k_tensor = torch.tensor(case[1], dtype=torch.float32)
                    elif dist_name == "Dirichlet":
                        distribution = Dirichlet(case[0])
                        k_tensor = case[1]  # Dirichlet expects tensor input
                    elif dist_name == "VonMises":
                        distribution = VonMises(torch.tensor(case[0]), torch.tensor(case[1]))
                        k_tensor = torch.tensor(case[2], dtype=torch.float32)
                    elif dist_name == "Uniform":
                        distribution = Uniform(torch.tensor(case[0]), torch.tensor(case[1]))
                        k_tensor = torch.tensor(case[2], dtype=torch.float32)
                    elif dist_name == "Chi2":
                        distribution = Chi2(torch.tensor(case[0]))  # df
                        k_tensor = torch.tensor(case[1], dtype=torch.float32)
                    else:
                        self.fail(f"Unhandled distribution: {dist_name}")

                    # Call wrapper log-prob method
                    if dist_name == "Dirichlet":
                        cdf = PtdmCdf().f(distribution, str(k_tensor.tolist()))[0]
                    else:
                        cdf = PtdmCdf().f(distribution, str(k_tensor.item()))[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(cdf, torch.Tensor), "Probability output is not a tensor")

                    # Convert tensors to floats for comparison
                    cdf = cdf.item()

                    # Check if computed values are approximately equal to expected values
                    self.assertAlmostEqual(cdf, case[-1], places=4, 
                                           msg=f"{dist_name}: Expected {case[-1]}, got {cdf}")

if __name__ == "__main__":
    unittest.main()
