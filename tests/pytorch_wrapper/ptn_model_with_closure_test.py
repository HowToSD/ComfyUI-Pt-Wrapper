import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_model_with_closure import PtnModelWithClosure


class TestPtnModelWithClosure(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test instance with dummy models."""
        self.node = PtnModelWithClosure()

        class ModelA(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(3, 3, bias=False)
                self.layer.weight = nn.Parameter(torch.eye(3), requires_grad=True)  # Identity

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layer(x)

        self.model_a = ModelA()
        self.closure = nn.ReLU()  # Apply ReLU as the closure

    def test_forward_pass(self) -> None:
        """Test forward pass through the chained model."""
        model, = self.node.f(self.model_a, self.closure)

        # Input tensor
        input_tensor = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)

        # Expected transformation:
        # ModelA (Identity):     [1, 0, -1]
        # Closure (ReLU):        [1, 0, 0]
        expected_output = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

        # Get output from the chained model
        output = model(input_tensor)

        # Ensure output is a tensor
        self.assertIsInstance(output, torch.Tensor, "Output is not a tensor")

        # Ensure output shape matches expected
        self.assertEqual(output.shape, expected_output.shape, "Output shape mismatch")

        # Ensure output values match expected
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6),
                        f"Expected {expected_output}, got {output}")

if __name__ == "__main__":
    unittest.main()
