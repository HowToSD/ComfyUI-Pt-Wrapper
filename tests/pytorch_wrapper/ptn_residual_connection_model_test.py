import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_residual_connection_model import PtnResidualConnectionModel


class TestPtnResidualConnectionModel(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test instance with dummy model."""
        self.node = PtnResidualConnectionModel()

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(3, 3, bias=False)
                self.layer.weight = nn.Parameter(torch.eye(3) * 0.2, requires_grad=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layer(x)

        self.model = DummyModel()
        self.closure = nn.ReLU()

    def test_forward_pass(self) -> None:
        """Test forward pass with residual addition and closure."""
        model, = self.node.f(self.model, self.closure)

        input_tensor = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)

        # model(input) = 0.2 * input = [0.2, 0.0, -0.2]
        # residual + model(input) = [1.2, 0.0, -1.2]
        # ReLU: [1.2, 0.0, 0.0]
        expected_output = torch.tensor([[1.2, 0.0, 0.0]], dtype=torch.float32)

        output = model(input_tensor)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6),
                        f"Expected {expected_output}, got {output}")

    def test_forward_pass_no_closure(self) -> None:
        """Test forward pass with residual addition without closure."""
        model, = self.node.f(self.model, None)

        input_tensor = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)

        # model(input) = 0.2 * input = [0.2, 0.0, -0.2]
        # residual + model(input) = [1.2, 0.0, -1.2]
        expected_output = torch.tensor([[1.2, 0.0, -1.2]], dtype=torch.float32)

        output = model(input_tensor)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6),
                        f"Expected {expected_output}, got {output}")


if __name__ == "__main__":
    unittest.main()
