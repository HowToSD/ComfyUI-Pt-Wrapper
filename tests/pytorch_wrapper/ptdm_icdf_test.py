import os
import sys
import unittest
import torch
import scipy.stats as scst
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptdm_icdf import PtdmIcdf
from torch.distributions import (
    Categorical, NegativeBinomial, Multinomial, 
    Gamma, Normal, Laplace, Exponential, LogNormal, 
    Pareto, Weibull, Dirichlet, VonMises, Uniform
)


class TestPtdfCdf(unittest.TestCase):
    def setUp(self):
        """Set up test instance.
        
        Following distributions are tested in separate unit test files:
        * Bernoulli
        * Binomial
        * Poisson
        * Chi2
        * Gamma
        * StudentT
        * Beta
        """
        self.test_cases = {
            "Uniform": [(0.0, 1.0, 0.5, scst.uniform.ppf(0.5, loc=0, scale=1))],
            "Normal": [(0.0, 1.0, 0.95, scst.norm.ppf(0.95, 0, 1))],
            "Exponential": [(1.0, 0.9, scst.expon.ppf(0.9, scale=1))],
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
                    else:
                        self.fail(f"Unhandled distribution: {dist_name}")

                    # Call wrapper log-prob method
                    if dist_name == "Dirichlet":
                        icdf = PtdmIcdf().f(distribution, str(k_tensor.tolist()))[0]
                    else:
                        icdf = PtdmIcdf().f(distribution, str(k_tensor.item()))[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(icdf, torch.Tensor), "Probability output is not a tensor")

                    # Convert tensors to floats for comparison
                    icdf = icdf.item()

                    # Check if computed values are approximately equal to expected values
                    self.assertAlmostEqual(icdf, case[-1], places=4, 
                                           msg=f"{dist_name}: Expected {case[-1]}, got {icdf}")

if __name__ == "__main__":
    unittest.main()
