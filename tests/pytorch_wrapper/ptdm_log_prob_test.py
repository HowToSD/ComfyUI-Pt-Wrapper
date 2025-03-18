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
from pytorch_wrapper.ptdm_log_prob import PtdmLogProb
from torch.distributions import (
    Binomial, Categorical, Poisson, NegativeBinomial, Multinomial, 
    Bernoulli, Beta, Gamma, Normal, Laplace, Exponential, LogNormal, 
    Pareto, Weibull, Dirichlet, VonMises, StudentT, Uniform
)


class TestPtdfLogProb(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.test_cases = {
            "Binomial": [(10, k, 0.5, np.log(scst.binom.pmf(k, 10, 0.5))) for k in range(11)],
            "Poisson": [(3.0, k, np.log(scst.poisson.pmf(k, 3.0))) for k in range(10)],
            "Categorical": [(torch.tensor([0.2, 0.8]), 0, np.log(0.2)), (torch.tensor([0.2, 0.8]), 1, np.log(0.8))],
            "NegativeBinomial": [(5, k, 0.5, np.log(scst.nbinom.pmf(k, 5, 0.5))) for k in range(10)],
            "Bernoulli": [(0.5, k, np.log(scst.bernoulli.pmf(k, 0.5))) for k in [0, 1]],
            "Beta": [(2.0, 5.0, 0.3, np.log(scst.beta.pdf(0.3, 2, 5)))],
            "Gamma": [(2.0, 1.0, 3.0, np.log(scst.gamma.pdf(3.0, 2, scale=1)))],
            "Normal": [(0.0, 1.0, -1.5, np.log(scst.norm.pdf(-1.5, 0, 1)))],
            "Laplace": [(0.0, 1.0, -1.5, np.log(scst.laplace.pdf(-1.5, 0, 1)))],
            "Exponential": [(1.0, 3.0, np.log(scst.expon.pdf(3.0, scale=1)))],
            "LogNormal": [(0.0, 1.0, 2.0, np.log(scst.lognorm.pdf(2.0, s=1, scale=np.exp(0))))],
            "Pareto": [(2.5, 3.0, 3.5, np.log(scst.pareto.pdf(3.5, b=3.0, scale=2.5)))],
            "Weibull": [(2.0, 3.0, np.log(scst.weibull_min.pdf(3.0, c=2.0, scale=1.0)))],
            "Dirichlet": [(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.2, 0.5, 0.3]), np.log(scst.dirichlet.pdf([0.2, 0.5, 0.3], [1.0, 1.0, 1.0])))],
            "VonMises": [(0.0, 1.0, 1.0, np.log(scst.vonmises.pdf(1.0, kappa=1.0, loc=0)))],
            "StudentT": [(1.0, 0.0, np.log(scst.t.pdf(0.0, 1)))],
            "Uniform": [(0.0, 1.0, 0.5, np.log(scst.uniform.pdf(0.5, loc=0, scale=1)))],
        }

    def test_distributions(self):
        """Test probability mass and density functions for all PyTorch distributions."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    if dist_name == "Binomial":
                        distribution = PtdBinomial().f(str(case[0]), str(case[2]), "")[0]  # n, p
                        log_prob = PtdmLogProb().f(distribution, str(case[1]))[0]  # k
                    else:
                        # Instantiate PyTorch distribution directly if not in wrapper
                        if dist_name == "Poisson":
                            distribution = Poisson(torch.tensor(case[0]))
                            k_tensor = torch.tensor(case[1], dtype=torch.int64)
                        elif dist_name == "Categorical":
                            distribution = Categorical(case[0])
                            k_tensor = torch.tensor(case[1], dtype=torch.int64)
                        elif dist_name == "NegativeBinomial":
                            distribution = NegativeBinomial(torch.tensor(case[0]), torch.tensor(case[2]))
                            k_tensor = torch.tensor(case[1], dtype=torch.int64)
                        elif dist_name == "Bernoulli":
                            distribution = Bernoulli(torch.tensor(case[0]))
                            k_tensor = torch.tensor(case[1], dtype=torch.float32)
                        elif dist_name == "Beta":
                            distribution = Beta(torch.tensor(case[0]), torch.tensor(case[1]))
                            k_tensor = torch.tensor(case[2], dtype=torch.float32)
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
                        elif dist_name == "StudentT":
                            distribution = StudentT(torch.tensor(case[0]))
                            k_tensor = torch.tensor(case[1], dtype=torch.float32)
                        elif dist_name == "Uniform":
                            distribution = Uniform(torch.tensor(case[0]), torch.tensor(case[1]))
                            k_tensor = torch.tensor(case[2], dtype=torch.float32)
                        else:
                            self.fail(f"Unhandled distribution: {dist_name}")

                        # Call wrapper log-prob method
                        if dist_name == "Dirichlet":
                            log_prob = PtdmLogProb().f(distribution, str(k_tensor.tolist()))[0]
                        else:
                            log_prob = PtdmLogProb().f(distribution, str(k_tensor.item()))[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(log_prob, torch.Tensor), "Probability output is not a tensor")

                    # Convert tensors to floats for comparison
                    log_prob = log_prob.item()

                    # Check if computed values are approximately equal to expected values
                    self.assertAlmostEqual(log_prob, case[-1], places=4, 
                                           msg=f"{dist_name}: Expected {case[-1]}, got {log_prob}")

if __name__ == "__main__":
    unittest.main()
