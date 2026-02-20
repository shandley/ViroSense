"""Tests for MLX Evo2 backend.

All tests use synthetic weights (no model download required).
Tests are skipped if MLX is not available.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from virosense.backends.mlx_model import (  # noqa: E402
    AttentionBlock,
    Evo2Config,
    Evo2Model,
    FIRFilter,
    GatedMLP,
    HyenaBlock,
    IIRFilter,
    RMSNorm,
    ShortFilter,
    _parse_layer_name,
    _permute_wqkv,
    fft_conv,
    tokenize_dna,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestEvo2Config:
    def test_default_config(self):
        config = Evo2Config()
        assert config.vocab_size == 512
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.inner_mlp_size == 11264
        assert config.state_size == 16

    def test_layer_assignments_cover_all_blocks(self):
        config = Evo2Config()
        all_layers = sorted(
            config.hcs_layers + config.hcm_layers + config.hcl_layers + config.mha_layers
        )
        assert all_layers == list(range(32))

    def test_layer_counts(self):
        config = Evo2Config()
        assert len(config.hcs_layers) == 9
        assert len(config.hcm_layers) == 9
        assert len(config.hcl_layers) == 9
        assert len(config.mha_layers) == 5


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_dna_to_tokens(self):
        tokens = tokenize_dna("ACGT")
        expected = mx.array([65, 67, 71, 84], dtype=mx.uint32)
        assert mx.array_equal(tokens, expected)

    def test_single_base(self):
        assert tokenize_dna("A").item() == 65
        assert tokenize_dna("C").item() == 67
        assert tokenize_dna("G").item() == 71
        assert tokenize_dna("T").item() == 84

    def test_n_base(self):
        assert tokenize_dna("N").item() == 78

    def test_empty_string(self):
        tokens = tokenize_dna("")
        assert tokens.shape == (0,)

    def test_long_sequence(self):
        seq = "ACGT" * 100
        tokens = tokenize_dna(seq)
        assert tokens.shape == (400,)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = mx.random.normal((2, 10, 64))
        y = norm(x)
        mx.eval(y)
        assert y.shape == (2, 10, 64)

    def test_normalization_magnitude(self):
        norm = RMSNorm(64)
        x = mx.random.normal((1, 10, 64)) * 10.0
        y = norm(x)
        mx.eval(y)
        # After normalization, RMS of each position should be ~1
        rms = np.sqrt(np.mean(np.array(y) ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=0.1)

    def test_scale_parameter(self):
        norm = RMSNorm(4)
        norm.scale = mx.array([2.0, 2.0, 2.0, 2.0])
        x = mx.ones((1, 1, 4))
        y = norm(x)
        mx.eval(y)
        # RMS of ones(4) = 1.0, so output should be 2.0 * 1.0
        np.testing.assert_allclose(np.array(y), 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# GatedMLP
# ---------------------------------------------------------------------------

class TestGatedMLP:
    def test_output_shape(self):
        mlp = GatedMLP(64, 128, use_gelu=False)
        x = mx.random.normal((2, 10, 64))
        y = mlp(x)
        mx.eval(y)
        assert y.shape == (2, 10, 64)

    def test_gelu_vs_identity(self):
        """GELU and Identity should produce different outputs for same input."""
        mx.random.seed(42)
        mlp_gelu = GatedMLP(32, 64, use_gelu=True)
        mlp_id = GatedMLP(32, 64, use_gelu=False)

        # Copy weights from gelu to id
        mlp_id.l1.weight = mlp_gelu.l1.weight
        mlp_id.l2.weight = mlp_gelu.l2.weight
        mlp_id.l3.weight = mlp_gelu.l3.weight

        x = mx.random.normal((1, 5, 32))
        y_gelu = mlp_gelu(x)
        y_id = mlp_id(x)
        mx.eval(y_gelu, y_id)

        # They should differ (unless l1 output is all zeros, which is unlikely)
        assert not mx.allclose(y_gelu, y_id).item()


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFFTConv:
    def test_output_shape(self):
        u = mx.random.normal((2, 64, 100))  # (B, D, L)
        h = mx.random.normal((64, 50))  # (D, K)
        y = fft_conv(u, h)
        mx.eval(y)
        assert y.shape == (2, 64, 100)

    def test_with_skip_connection(self):
        u = mx.random.normal((1, 16, 50))
        h = mx.random.normal((16, 30))
        d = mx.random.normal((16,))
        y = fft_conv(u, h, d)
        mx.eval(y)
        assert y.shape == (1, 16, 50)

    def test_grouped_filter(self):
        u = mx.random.normal((1, 256, 100))
        h = mx.random.normal((64, 1, 20))  # grouped: 64 groups for 256 channels
        y = fft_conv(u, h)
        mx.eval(y)
        assert y.shape == (1, 256, 100)


class TestShortFilter:
    def test_output_shape(self):
        sf = ShortFilter(192, kernel_size=3)  # 3 * 64
        x = mx.random.normal((2, 50, 192))  # (B, L, C) NLC
        y = sf(x)
        mx.eval(y)
        assert y.shape == (2, 50, 192)

    def test_causal(self):
        """Output at position t should not depend on future inputs."""
        sf = ShortFilter(4, kernel_size=3)
        sf.weight = mx.ones((4, 1, 3))

        x = mx.zeros((1, 10, 4))
        # Set a spike at position 5
        x_np = np.zeros((1, 10, 4))
        x_np[0, 5, :] = 1.0
        x = mx.array(x_np)

        y = sf(x)
        mx.eval(y)
        y_np = np.array(y)

        # Positions before 5 should be zero (causal: no future leakage)
        np.testing.assert_allclose(y_np[0, :5, :], 0.0, atol=1e-6)
        # Position 5+ should have nonzero values
        assert np.any(np.abs(y_np[0, 5:, :]) > 0.1)


class TestFIRFilter:
    def test_hcs_short_output_shape(self):
        fir = FIRFilter(filter_length=7, filter_groups=256, hidden_size=4096, has_d=False)
        u = mx.random.normal((1, 4096, 100))
        y = fir(u)
        mx.eval(y)
        assert y.shape == (1, 4096, 100)

    def test_hcm_medium_output_shape(self):
        fir = FIRFilter(filter_length=128, filter_groups=256, hidden_size=4096, has_d=True)
        u = mx.random.normal((1, 4096, 200))
        y = fir(u)
        mx.eval(y)
        assert y.shape == (1, 4096, 200)

    def test_no_d_means_no_skip(self):
        fir = FIRFilter(filter_length=7, filter_groups=4, hidden_size=8, has_d=False)
        assert not hasattr(fir, "D") or not fir.has_d


class TestIIRFilter:
    def test_compute_filter_shape(self):
        iir = IIRFilter(hidden_size=64, state_size=16)
        # Set negative log_poles for stability
        iir.log_poles = mx.full((64, 16, 1), -0.1)
        iir.residues = mx.random.normal((64, 16))
        h = iir.compute_filter(100)
        mx.eval(h)
        assert h.shape == (1, 64, 100)

    def test_filter_decays(self):
        """With negative log_poles, filter should decay over time."""
        iir = IIRFilter(hidden_size=4, state_size=2)
        iir.log_poles = mx.full((4, 2, 1), -0.5)
        iir.residues = mx.ones((4, 2))
        h = iir.compute_filter(50)
        mx.eval(h)
        h_np = np.array(h)
        # Filter value at t=0 should be larger than at t=49
        assert np.all(np.abs(h_np[0, :, 0]) > np.abs(h_np[0, :, 49]))

    def test_forward_output_shape(self):
        iir = IIRFilter(hidden_size=64, state_size=16)
        iir.log_poles = mx.full((64, 16, 1), -0.1)
        iir.residues = mx.random.normal((64, 16))
        iir.D = mx.zeros((64,))
        u = mx.random.normal((1, 64, 100))
        y = iir(u)
        mx.eval(y)
        assert y.shape == (1, 64, 100)


# ---------------------------------------------------------------------------
# Blocks (tiny config for fast tests)
# ---------------------------------------------------------------------------

def _tiny_config() -> Evo2Config:
    """Minimal config for fast tests."""
    return Evo2Config(
        vocab_size=32,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        head_dim=16,
        inner_mlp_size=128,
        state_size=4,
        hcs_layers=[0],
        hcm_layers=[1],
        hcl_layers=[2],
        mha_layers=[3],
        hcs_filter_length=3,
        hcs_filter_groups=16,
        hcm_filter_length=8,
        hcm_filter_groups=16,
        hcl_filter_groups=64,
    )


class TestHyenaBlock:
    def test_hcs_output_shape(self):
        config = _tiny_config()
        block = HyenaBlock(config, layer_idx=0)  # HCS
        x = mx.random.normal((1, 20, 64))
        y = block(x)
        mx.eval(y)
        assert y.shape == (1, 20, 64)

    def test_hcm_output_shape(self):
        config = _tiny_config()
        block = HyenaBlock(config, layer_idx=1)  # HCM
        x = mx.random.normal((1, 20, 64))
        y = block(x)
        mx.eval(y)
        assert y.shape == (1, 20, 64)

    def test_hcl_output_shape(self):
        config = _tiny_config()
        block = HyenaBlock(config, layer_idx=2)  # HCL
        # Set negative log_poles for stability
        block.inner_filter.log_poles = mx.full((64, 4, 1), -0.1)
        x = mx.random.normal((1, 20, 64))
        y = block(x)
        mx.eval(y)
        assert y.shape == (1, 20, 64)

    def test_residual_connection(self):
        """Output should differ from input (not just residual pass-through)."""
        config = _tiny_config()
        block = HyenaBlock(config, layer_idx=0)
        x = mx.random.normal((1, 10, 64))
        y = block(x)
        mx.eval(y)
        assert not mx.allclose(x, y).item()


class TestAttentionBlock:
    def test_output_shape(self):
        config = _tiny_config()
        block = AttentionBlock(config, layer_idx=3)
        x = mx.random.normal((1, 20, 64))
        y = block(x)
        mx.eval(y)
        assert y.shape == (1, 20, 64)

    def test_batch_dimension(self):
        config = _tiny_config()
        block = AttentionBlock(config, layer_idx=3)
        x = mx.random.normal((3, 15, 64))
        y = block(x)
        mx.eval(y)
        assert y.shape == (3, 15, 64)


# ---------------------------------------------------------------------------
# Full Model (tiny)
# ---------------------------------------------------------------------------

class TestEvo2Model:
    def test_forward_shape_batched(self):
        config = _tiny_config()
        model = Evo2Model(config)
        # Fix IIR log_poles for stability
        for block in model.blocks:
            if isinstance(block, HyenaBlock) and isinstance(block.inner_filter, IIRFilter):
                block.inner_filter.log_poles = mx.full(
                    block.inner_filter.log_poles.shape, -0.1
                )
        tokens = mx.array([[65, 67, 71, 84, 65]], dtype=mx.uint32)  # ACGTA
        y = model(tokens, extract_layer="blocks.3.mlp.l3")
        mx.eval(y)
        assert y.shape == (1, 64)  # (B, D)

    def test_forward_shape_unbatched(self):
        config = _tiny_config()
        model = Evo2Model(config)
        for block in model.blocks:
            if isinstance(block, HyenaBlock) and isinstance(block.inner_filter, IIRFilter):
                block.inner_filter.log_poles = mx.full(
                    block.inner_filter.log_poles.shape, -0.1
                )
        tokens = mx.array([65, 67, 71, 84, 65], dtype=mx.uint32)
        y = model(tokens, extract_layer="blocks.3.mlp.l3")
        mx.eval(y)
        assert y.shape == (64,)  # (D,) — no batch dim

    def test_different_layers_different_embeddings(self):
        config = _tiny_config()
        model = Evo2Model(config)
        for block in model.blocks:
            if isinstance(block, HyenaBlock) and isinstance(block.inner_filter, IIRFilter):
                block.inner_filter.log_poles = mx.full(
                    block.inner_filter.log_poles.shape, -0.1
                )
        tokens = mx.array([65, 67, 71, 84], dtype=mx.uint32)
        y0 = model(tokens, extract_layer="blocks.0.mlp.l3")
        y3 = model(tokens, extract_layer="blocks.3.mlp.l3")
        mx.eval(y0, y3)
        assert not mx.allclose(y0, y3).item()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

class TestParseLayerName:
    def test_valid(self):
        block, sub = _parse_layer_name("blocks.28.mlp.l3")
        assert block == 28
        assert sub == "mlp.l3"

    def test_different_block(self):
        block, sub = _parse_layer_name("blocks.0.mlp.l1")
        assert block == 0
        assert sub == "mlp.l1"

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid layer name"):
            _parse_layer_name("decoder.layers.28")


class TestPermuteWqkv:
    def test_permutation_shape(self):
        num_heads = 4
        head_dim = 16
        hidden = 64
        w = mx.random.normal((3 * hidden, hidden))
        out = _permute_wqkv(w, num_heads, head_dim)
        mx.eval(out)
        assert out.shape == (3 * hidden, hidden)

    def test_permutation_correctness(self):
        """Verify Q, K, V are correctly separated after permutation."""
        num_heads = 2
        head_dim = 4
        hidden = 8

        # Create known pattern: per-head interleaved [H0_Q, H0_K, H0_V, H1_Q, H1_K, H1_V]
        w = mx.zeros((24, hidden))  # 3 * 8 = 24
        # H0_Q (rows 0-3), H0_K (4-7), H0_V (8-11), H1_Q (12-15), H1_K (16-19), H1_V (20-23)
        w_np = np.zeros((24, hidden), dtype=np.float32)
        w_np[0:4] = 1.0    # H0_Q
        w_np[4:8] = 2.0    # H0_K
        w_np[8:12] = 3.0   # H0_V
        w_np[12:16] = 1.0  # H1_Q
        w_np[16:20] = 2.0  # H1_K
        w_np[20:24] = 3.0  # H1_V
        w = mx.array(w_np)

        out = _permute_wqkv(w, num_heads, head_dim)
        mx.eval(out)
        out_np = np.array(out)

        # After permutation: [Q_all, K_all, V_all]
        q = out_np[:8]   # Q for all heads
        k = out_np[8:16]  # K for all heads
        v = out_np[16:24] # V for all heads

        np.testing.assert_allclose(q, 1.0)
        np.testing.assert_allclose(k, 2.0)
        np.testing.assert_allclose(v, 3.0)


# ---------------------------------------------------------------------------
# MLXBackend
# ---------------------------------------------------------------------------

class TestMLXBackend:
    def test_is_available_without_weights(self, tmp_path):
        from virosense.backends.mlx_backend import MLXBackend
        backend = MLXBackend(model_dir=str(tmp_path / "nonexistent"))
        assert not backend.is_available()

    def test_max_context_length(self):
        from virosense.backends.mlx_backend import MLXBackend
        backend = MLXBackend()
        assert backend.max_context_length() == 131_072

    def test_sanitize_sequences_basic(self):
        from virosense.backends.mlx_backend import MLXBackend
        seqs = {"seq1": "acgt", "seq2": "ACGT"}
        sanitized = MLXBackend._sanitize_sequences(seqs)
        assert sanitized["seq1"] == "ACGT"
        assert sanitized["seq2"] == "ACGT"

    def test_sanitize_sequences_n_replacement(self):
        from virosense.backends.mlx_backend import MLXBackend
        seqs = {"seq1": "ACNGT"}
        sanitized = MLXBackend._sanitize_sequences(seqs)
        assert "N" not in sanitized["seq1"]
        assert len(sanitized["seq1"]) == 5

    def test_sanitize_sequences_invalid_chars(self):
        from virosense.backends.mlx_backend import MLXBackend
        with pytest.raises(ValueError, match="invalid characters"):
            MLXBackend._sanitize_sequences({"seq1": "ACGX"})

    def test_sanitize_sequences_empty(self):
        from virosense.backends.mlx_backend import MLXBackend
        with pytest.raises(ValueError, match="empty"):
            MLXBackend._sanitize_sequences({"seq1": ""})

    def test_factory_registration(self):
        """Verify 'mlx' is registered in the backend factory."""
        from virosense.backends.base import get_backend
        # Should not raise — just instantiate (model loading is lazy)
        backend = get_backend("mlx", model_dir="/tmp/nonexistent")
        assert backend.max_context_length() == 131_072

    def test_factory_unknown_backend(self):
        from virosense.backends.base import get_backend
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")
