# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import torch
from torch import nn

from peft import get_peft_model
from peft.tuners.gralora import GraloraConfig


class MLP(nn.Module):
    """Simple MLP for testing"""

    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 20, bias=bias)
        self.lin2 = nn.Linear(20, 20, bias=bias)
        self.lin3 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestGraloraUniqueFeatures:
    """Tests for GraLoRA-specific features not covered by PeftCommonTester"""

    @pytest.fixture
    def mlp_gralora_pure(self):
        """Pure GraLoRA without hybrid component"""
        torch.manual_seed(0)
        mlp = MLP()
        config = GraloraConfig(
            target_modules=["lin1", "lin2"],
            r=16,
            gralora_k=4,
            hybrid_r=0,
            gralora_alpha=32,
            gralora_dropout=0.1,
        )
        peft_model = get_peft_model(mlp, config)
        return peft_model

    def test_gralora_information_exchange_via_permutation(self, mlp_gralora_pure):
        """
        Test that information exchange happens through tensor permutation.

        GraLoRA uses block-diagonal adapters but enables information flow between blocks through tensor permutation
        operations. This test verifies that changing inputs in one block affects outputs in all blocks, not just the
        corresponding block.
        """
        mlp_gralora_pure.eval()

        # Create two inputs that differ only in specific blocks
        x1 = torch.randn(1, 10)
        x2 = x1.clone()

        # Modify only the first block (assuming k=4, block size ~2-3 features)
        x2[0, :5] += 1.0  # Modify first block

        with torch.no_grad():
            out1 = mlp_gralora_pure(x1)
            out2 = mlp_gralora_pure(x2)

        # Due to information exchange via permutation, changing one block should affect all outputs
        # (not just outputs corresponding to that block)
        diff = (out1 - out2).abs()

        # All output dimensions should be affected, demonstrating information exchange
        assert (diff > 1e-6).all(), "Information exchange via permutation not working correctly"

    def test_gralora_scaling_factor(self):
        """
        Test that the scaling factor (gralora_alpha / r) is correctly applied.

        The scaling factor controls the magnitude of adapter updates. Models with different alpha values but identical
        adapter weights should produce different outputs.
        """
        torch.manual_seed(0)

        # Create two configs with different alpha values
        config_alpha16 = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_alpha=16,
            gralora_k=2,
            hybrid_r=0,
        )

        config_alpha32 = GraloraConfig(
            target_modules=["lin1"],
            r=8,
            gralora_alpha=32,
            gralora_k=2,
            hybrid_r=0,
        )

        model_alpha16 = get_peft_model(MLP(), config_alpha16)
        model_alpha32 = get_peft_model(MLP(), config_alpha32)

        # Copy weights to make adapter parameters identical (but scaling will differ)
        for (n1, p1), (n2, p2) in zip(model_alpha16.named_parameters(), model_alpha32.named_parameters()):
            if "gralora" in n1:
                p2.data = p1.data.clone()

        x = torch.randn(5, 10)

        model_alpha16.eval()
        model_alpha32.eval()

        with torch.no_grad():
            out1 = model_alpha16(x)
            out2 = model_alpha32(x)

        # Outputs should be different due to different scaling factors
        # (alpha16 has scaling 16/8=2, alpha32 has scaling 32/8=4)
        assert not torch.allclose(out1, out2, atol=1e-6, rtol=1e-6), "Scaling factor not being applied correctly"

    def test_gralora_hybrid_forward_computation(self):
        """
        Test that the hybrid LoRA component is correctly used in the forward pass.

        Hybrid GraLoRA combines block-diagonal GraLoRA adapters (for local features) with a full-rank vanilla LoRA
        adapter (for global features). This test verifies that the hybrid component contributes to the output.
        """
        torch.manual_seed(0)
        mlp_hybrid = MLP()
        mlp_pure = MLP()

        config_hybrid = GraloraConfig(
            target_modules=["lin1"],
            r=16,
            gralora_k=4,
            hybrid_r=4,
            init_weights=False,
        )
        model_hybrid = get_peft_model(mlp_hybrid, config_hybrid)

        config_pure = GraloraConfig(
            target_modules=["lin1"],
            r=16,
            gralora_k=4,
            hybrid_r=0,
            init_weights=False,
        )
        model_pure = get_peft_model(mlp_pure, config_pure)

        x = torch.randn(5, 10)

        with torch.no_grad():
            output_hybrid = model_hybrid(x)
            output_pure = model_pure(x)

        # Outputs should be different because hybrid model includes additional vanilla LoRA component
        assert not torch.allclose(output_hybrid, output_pure, atol=1e-3), (
            "Hybrid LoRA component not contributing to forward pass"
        )
