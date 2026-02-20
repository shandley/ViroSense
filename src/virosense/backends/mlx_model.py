"""Evo2 7B model implemented in Apple MLX for embedding extraction.

This is a forward-pass-only implementation of StripedHyena 2 (Evo2 7B)
targeting Apple Silicon. No generation, no KV cache — only embedding
extraction via mean-pooling of intermediate hidden states.

Architecture: 32 blocks (9 HCS + 9 HCM + 9 HCL + 5 MHA) with
hidden_size=4096, vocab_size=512 (byte-level tokenizer).

Reference: https://github.com/ArcInstitute/evo2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Evo2Config:
    """Evo2 7B hyperparameters."""

    vocab_size: int = 512
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    inner_mlp_size: int = 11264
    state_size: int = 16  # IIR filter order
    eps: float = 1e-6  # RMSNorm epsilon
    short_filter_length: int = 3  # outer short filter on all Hyena blocks
    rope_theta: float = 10_000.0

    # Layer assignments (0-indexed block ids)
    hcs_layers: list[int] = field(
        default_factory=lambda: [0, 4, 7, 11, 14, 18, 21, 25, 28]
    )
    hcm_layers: list[int] = field(
        default_factory=lambda: [1, 5, 8, 12, 15, 19, 22, 26, 29]
    )
    hcl_layers: list[int] = field(
        default_factory=lambda: [2, 6, 9, 13, 16, 20, 23, 27, 30]
    )
    mha_layers: list[int] = field(
        default_factory=lambda: [3, 10, 17, 24, 31]
    )

    # Filter parameters
    hcs_filter_length: int = 7
    hcs_filter_groups: int = 256
    hcm_filter_length: int = 128
    hcm_filter_groups: int = 256
    hcl_filter_groups: int = 4096  # one per channel


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize_dna(sequence: str) -> mx.array:
    """Convert DNA string to byte-level token IDs.

    Evo2 uses a byte-level tokenizer where each character maps to its
    UTF-8 byte value. For DNA: A=65, C=67, G=71, T=84.

    Args:
        sequence: Uppercase DNA string (ACGTN only).

    Returns:
        1D mx.array of uint32 token IDs, shape (L,).
    """
    return mx.array(list(sequence.encode("utf-8")), dtype=mx.uint32)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Matches Evo2's implementation: x / (RMS(x) + eps) * scale
    where RMS(x) = sqrt(mean(x^2)).
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.scale = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms * self.scale).astype(dtype)


class GatedMLP(nn.Module):
    """Gated MLP with optional GELU activation.

    Evo2 quirk: only layer 0 uses GELU. Layers 1-31 use identity
    (evo2_style_activations=True), making this a pure gated linear unit.
    """

    def __init__(self, hidden_size: int, inner_size: int, use_gelu: bool = False):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, hidden_size, bias=False)
        self.use_gelu = use_gelu

    def __call__(self, x: mx.array) -> mx.array:
        z1 = self.l1(x)
        z2 = self.l2(x)
        if self.use_gelu:
            z1 = nn.gelu(z1)
        return self.l3(z1 * z2)


# ---------------------------------------------------------------------------
# Filters (Hyena convolution operators)
# ---------------------------------------------------------------------------

def fft_conv(u: mx.array, h: mx.array, d: mx.array | None = None) -> mx.array:
    """FFT-based 1D convolution.

    Args:
        u: Input signal, shape (B, D, L) in float32.
        h: Filter kernel, shape (D, L) or (G, 1, K) where G divides D.
        d: Optional skip connection bias, shape (D,).

    Returns:
        Convolved signal, shape (B, D, L).
    """
    L = u.shape[-1]
    n = 2 * L

    # Ensure h is 2D: (D_or_G, K)
    if h.ndim == 3:
        h = h.squeeze(1)  # (G, K) — remove middle dim
    K = h.shape[-1]

    # Pad h to match signal length if needed, then do FFT conv
    if K < L:
        h_padded = mx.zeros((h.shape[0], L), dtype=h.dtype)
        h_padded[:, :K] = h
        h = h_padded

    H = mx.fft.rfft(h.astype(mx.float32), n=n, axis=-1)
    U = mx.fft.rfft(u.astype(mx.float32), n=n, axis=-1)

    # Broadcast: H is (G, n//2+1), U is (B, D, n//2+1)
    # If G < D (grouped conv), repeat H to match D
    if H.shape[0] < U.shape[1]:
        repeats = U.shape[1] // H.shape[0]
        H = mx.repeat(H, repeats, axis=0)

    y = mx.fft.irfft(U * H[None], n=n, axis=-1)[..., :L]

    if d is not None:
        y = y + u * d[None, :, None]

    return y


class ShortFilter(nn.Module):
    """Depthwise conv1d with kernel=3 applied to all 3*hidden channels.

    This is the outer short filter present in all Hyena blocks.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        # Weight shape: (channels, 1, kernel_size) for depthwise conv
        self.weight = mx.zeros((channels, 1, kernel_size))
        self.channels = channels
        self.kernel_size = kernel_size

    def __call__(self, x: mx.array) -> mx.array:
        """Apply depthwise conv1d.

        Args:
            x: Input shape (B, L, C) in NLC layout.

        Returns:
            Output shape (B, L, C).
        """
        B, L, C = x.shape
        # Transpose to NCL for manual FFT conv, then back
        x_ncl = mx.transpose(x, axes=(0, 2, 1))  # (B, C, L)

        # Use direct conv via padding + matmul for short kernels
        pad = self.kernel_size - 1
        x_padded = mx.pad(x_ncl, [(0, 0), (0, 0), (pad, 0)])  # left-pad (causal)
        # Sliding window via as_strided would be ideal but we'll use a simple loop-free approach
        # For kernel_size=3, this is: w0*x[t-2] + w1*x[t-1] + w2*x[t]
        out = mx.zeros_like(x_ncl)
        for k in range(self.kernel_size):
            offset = pad - k
            out = out + x_padded[:, :, offset:offset + L] * self.weight[:, 0, k:k + 1]

        return mx.transpose(out, axes=(0, 2, 1))  # back to NLC


class FIRFilter(nn.Module):
    """Explicit FIR filter for HCS (k=7) and HCM (k=128) blocks.

    Short filters use direct depthwise conv1d. Medium filters (k>=128)
    use FFT convolution for efficiency.
    """

    def __init__(self, filter_length: int, filter_groups: int, hidden_size: int,
                 has_d: bool = False):
        super().__init__()
        self.h = mx.zeros((filter_groups, 1, filter_length))
        self.filter_length = filter_length
        self.filter_groups = filter_groups
        self.hidden_size = hidden_size
        self.has_d = has_d
        if has_d:
            self.D = mx.zeros((hidden_size,))

    def __call__(self, u: mx.array) -> mx.array:
        """Apply FIR filter.

        Args:
            u: Input shape (B, D, L) in NCL layout.

        Returns:
            Filtered signal shape (B, D, L).
        """
        d = self.D if self.has_d else None

        if self.filter_length < 128:
            # Short FIR: direct depthwise conv
            L = u.shape[-1]
            pad = self.filter_length - 1
            u_padded = mx.pad(u, [(0, 0), (0, 0), (pad, 0)])

            h = self.h.squeeze(1)  # (G, K)
            # Repeat filter groups to match channels
            if self.filter_groups < self.hidden_size:
                repeats = self.hidden_size // self.filter_groups
                h = mx.repeat(h, repeats, axis=0)  # (D, K)

            out = mx.zeros_like(u)
            for k in range(self.filter_length):
                offset = pad - k
                out = out + u_padded[:, :, offset:offset + L] * h[:, k:k + 1][None]

            if d is not None:
                out = out + u * d[None, :, None]
            return out
        else:
            # Medium/long FIR: FFT convolution
            return fft_conv(u, self.h, d)


class IIRFilter(nn.Module):
    """Implicit IIR filter for HCL (long) blocks.

    Computes filter kernel from log-space poles and residues via
    h(t) = sum_s residues_s * exp(log_poles_s * t), then applies
    via FFT convolution.
    """

    def __init__(self, hidden_size: int, state_size: int):
        super().__init__()
        # log_poles: (hidden_size, state_size, 1) — real-valued, negative for decay
        self.log_poles = mx.zeros((hidden_size, state_size, 1))
        # residues: (hidden_size, state_size) — real-valued
        self.residues = mx.zeros((hidden_size, state_size))
        self.D = mx.zeros((hidden_size,))
        self.hidden_size = hidden_size
        self.state_size = state_size

    def compute_filter(self, L: int) -> mx.array:
        """Compute IIR filter kernel of length L.

        Returns:
            Filter kernel shape (1, hidden_size, L).
        """
        t = mx.arange(L, dtype=mx.float32).reshape(1, 1, L)  # (1, 1, L)
        # log_poles: (D, S, 1), t: (1, 1, L) -> (D, S, L)
        decay = mx.exp(self.log_poles * t)
        # residues: (D, S) -> (D, S, 1) * (D, S, L) -> sum over S -> (D, L)
        h = (self.residues[..., None] * decay).sum(axis=1)
        return h[None]  # (1, D, L)

    def __call__(self, u: mx.array) -> mx.array:
        """Apply IIR filter via FFT convolution.

        Args:
            u: Input shape (B, D, L) in NCL layout.

        Returns:
            Filtered signal shape (B, D, L).
        """
        L = u.shape[-1]
        h = self.compute_filter(L)  # (1, D, L)
        h = h.squeeze(0)  # (D, L)

        return fft_conv(u, h, self.D)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class HyenaBlock(nn.Module):
    """Parallel Gated Convolution Block (Hyena operator).

    Used for HCS, HCM, and HCL layers. The filter type varies:
    - HCS: FIR with k=7
    - HCM: FIR with k=128 (FFT conv)
    - HCL: IIR from poles/residues (FFT conv)
    """

    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__()
        D = config.hidden_size

        self.pre_norm = RMSNorm(D, config.eps)
        self.post_norm = RMSNorm(D, config.eps)

        # Input projection: hidden -> 3*hidden (x2, x1, v)
        self.projections = nn.Linear(D, 3 * D, bias=False)

        # Outer short filter (kernel=3, depthwise on 3*hidden channels)
        self.short_filter = ShortFilter(3 * D, config.short_filter_length)

        # Inner filter (type depends on layer assignment)
        if layer_idx in config.hcs_layers:
            self.inner_filter = FIRFilter(
                config.hcs_filter_length, config.hcs_filter_groups, D, has_d=False
            )
        elif layer_idx in config.hcm_layers:
            self.inner_filter = FIRFilter(
                config.hcm_filter_length, config.hcm_filter_groups, D, has_d=True
            )
        elif layer_idx in config.hcl_layers:
            self.inner_filter = IIRFilter(D, config.state_size)
        else:
            raise ValueError(f"Layer {layer_idx} is not a Hyena layer")

        # Output projection
        self.out_filter_dense = nn.Linear(D, D, bias=True)

        # MLP (GELU only on layer 0)
        self.mlp = GatedMLP(D, config.inner_mlp_size, use_gelu=(layer_idx == 0))

        self.layer_idx = layer_idx
        self.hidden_size = D

    def __call__(self, u: mx.array) -> mx.array:
        """Forward pass.

        Args:
            u: Input shape (B, L, D) in NLC layout.

        Returns:
            Output shape (B, L, D).
        """
        residual = u
        z = self.pre_norm(u)

        # Project to 3*hidden
        z = self.projections(z)  # (B, L, 3D)

        # Short filter (depthwise conv on all 3*hidden channels)
        z = self.short_filter(z)  # (B, L, 3D)

        # Interleave reorder: [x_0, y_0, v_0, x_1, y_1, v_1, ...]
        # -> [x_0, x_1, ..., y_0, y_1, ..., v_0, v_1, ...]
        B, L, C = z.shape
        D = self.hidden_size
        z = z.reshape(B, L, D, 3)
        z = mx.transpose(z, axes=(0, 1, 3, 2))  # (B, L, 3, D)
        z = z.reshape(B, L, 3 * D)

        # Split into x2 (gate), x1, v
        x2 = z[:, :, :D]
        x1 = z[:, :, D:2 * D]
        v = z[:, :, 2 * D:]

        # Transpose to NCL for convolution operations
        x1_ncl = mx.transpose(x1, axes=(0, 2, 1))  # (B, D, L)
        v_ncl = mx.transpose(v, axes=(0, 2, 1))  # (B, D, L)

        # Gated convolution: filter(x1 * v) * x2
        x1v = x1_ncl * v_ncl  # (B, D, L)
        filtered = self.inner_filter(x1v)  # (B, D, L)

        # Gate with x2
        filtered_nlc = mx.transpose(filtered, axes=(0, 2, 1))  # (B, L, D)
        y = filtered_nlc * x2  # (B, L, D)

        # Output projection + residual
        y = self.out_filter_dense(y) + residual

        # MLP + residual
        z_mlp = self.post_norm(y)
        y = self.mlp(z_mlp) + y

        return y


class AttentionBlock(nn.Module):
    """Multi-Head Attention block with RoPE.

    Used at layers [3, 10, 17, 24, 31] in the 7B model.
    Standard attention: Wqkv -> RoPE -> SDPA -> out_proj.
    """

    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__()
        D = config.hidden_size

        self.pre_norm = RMSNorm(D, config.eps)
        self.post_norm = RMSNorm(D, config.eps)

        # Fused QKV projection
        self.Wqkv = nn.Linear(D, 3 * D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=True)

        # Rotary position embedding
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)

        # MLP (GELU only on layer 0 — but layer 0 is always HCS, so this is always False)
        self.mlp = GatedMLP(D, config.inner_mlp_size, use_gelu=(layer_idx == 0))

        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input shape (B, L, D).

        Returns:
            Output shape (B, L, D).
        """
        residual = x
        h = self.pre_norm(x)

        # QKV projection
        qkv = self.Wqkv(h)  # (B, L, 3*D)
        B, L, _ = qkv.shape

        # Split and reshape to (B, num_heads, L, head_dim)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0]  # (B, L, num_heads, head_dim)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Transpose to (B, num_heads, L, head_dim) for SDPA
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        # Scaled dot-product attention (causal mask for consistency with training)
        scale = self.head_dim ** -0.5
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=None  # no causal mask for embedding extraction
        )

        # Reshape back to (B, L, D)
        attn_out = mx.transpose(attn_out, axes=(0, 2, 1, 3))  # (B, L, H, d)
        attn_out = attn_out.reshape(B, L, -1)  # (B, L, D)

        # Output projection + residual
        y = self.out_proj(attn_out) + residual

        # MLP + residual
        z_mlp = self.post_norm(y)
        y = self.mlp(z_mlp) + y

        return y


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class Evo2Model(nn.Module):
    """Evo2 7B model for embedding extraction.

    Forward pass: embedding -> 32 blocks -> final norm.
    Extracts mean-pooled hidden states from a specified layer.
    """

    def __init__(self, config: Evo2Config | None = None):
        super().__init__()
        if config is None:
            config = Evo2Config()
        self.config = config

        # Embedding
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)

        # Build blocks
        self.blocks: list[HyenaBlock | AttentionBlock] = []
        mha_set = set(config.mha_layers)
        for i in range(config.num_layers):
            if i in mha_set:
                self.blocks.append(AttentionBlock(config, i))
            else:
                self.blocks.append(HyenaBlock(config, i))

        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.eps)

    def __call__(
        self,
        input_ids: mx.array,
        extract_layer: str = "blocks.28.mlp.l3",
    ) -> mx.array:
        """Forward pass with intermediate layer extraction.

        Args:
            input_ids: Token IDs, shape (B, L) or (L,).
            extract_layer: Layer name to extract embeddings from.
                Format: "blocks.{i}.mlp.l3" for MLP output of block i.

        Returns:
            Mean-pooled embedding, shape (B, D) or (D,) if unbatched.
        """
        squeeze = False
        if input_ids.ndim == 1:
            input_ids = input_ids[None]  # add batch dim
            squeeze = True

        # Parse target layer
        target_block, target_sublayer = _parse_layer_name(extract_layer)

        # Embedding
        x = self.embedding_layer(input_ids)  # (B, L, D)

        # Forward through blocks
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Extract from this block if it's the target
            if i == target_block:
                embedding = _extract_sublayer(x, block, target_sublayer)
                if squeeze:
                    return embedding.squeeze(0)
                return embedding

        # If we get here, extract from final norm (shouldn't happen for valid layer names)
        x = self.norm(x)
        embedding = x.mean(axis=1)  # (B, D)
        if squeeze:
            return embedding.squeeze(0)
        return embedding


def _parse_layer_name(name: str) -> tuple[int, str]:
    """Parse 'blocks.28.mlp.l3' into (28, 'mlp.l3')."""
    parts = name.split(".")
    if parts[0] != "blocks" or len(parts) < 3:
        raise ValueError(
            f"Invalid layer name: {name!r}. Expected 'blocks.{{i}}.sublayer'"
        )
    block_idx = int(parts[1])
    sublayer = ".".join(parts[2:])
    return block_idx, sublayer


def _extract_sublayer(
    block_output: mx.array,
    block: HyenaBlock | AttentionBlock,
    sublayer: str,
) -> mx.array:
    """Extract and mean-pool from a specific sublayer.

    For 'mlp.l3', we use the full block output (which includes the MLP).
    Mean-pool across the sequence dimension.
    """
    # The block output already includes the MLP pass, so for "mlp.l3"
    # (the most common extraction point), we just mean-pool the output.
    # More granular extraction (e.g., pre-MLP, specific MLP sublayer)
    # would require modifying the block forward pass.
    return block_output.mean(axis=1)  # (B, D)


# ---------------------------------------------------------------------------
# Weight Loading
# ---------------------------------------------------------------------------

def load_weights(model: Evo2Model, model_dir: str | Path) -> None:
    """Load safetensors weights into the MLX model.

    Handles key remapping from the HuggingFace checkpoint format
    (with 'backbone.' prefix) to our module structure.

    Args:
        model: Evo2Model instance to load weights into.
        model_dir: Directory containing model.safetensors.
    """
    import mlx.core as mx

    model_dir = Path(model_dir)
    weight_path = model_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weight_path}. "
            f"Run: python scripts/download_evo2_weights.py"
        )

    logger.info(f"Loading Evo2 7B weights from {weight_path}")

    # Load all tensors using MLX's native loader (handles bfloat16)
    weights = mx.load(str(weight_path))

    # Map weights to model
    config = model.config
    mha_set = set(config.mha_layers)

    # Global weights
    model.embedding_layer.weight = weights["backbone.embedding_layer.weight"]
    model.norm.scale = weights["backbone.norm.scale"]

    for i in range(config.num_layers):
        prefix = f"backbone.blocks.{i}"
        block = model.blocks[i]

        # Norms
        block.pre_norm.scale = weights[f"{prefix}.pre_norm.scale"]
        block.post_norm.scale = weights[f"{prefix}.post_norm.scale"]

        # MLP (shared by all block types)
        block.mlp.l1.weight = weights[f"{prefix}.mlp.l1.weight"]
        block.mlp.l2.weight = weights[f"{prefix}.mlp.l2.weight"]
        block.mlp.l3.weight = weights[f"{prefix}.mlp.l3.weight"]

        if i in mha_set:
            # Attention block
            wqkv = weights[f"{prefix}.inner_mha_cls.Wqkv.weight"]
            # Permute from per-head interleaved [num_heads, 3, head_dim, D]
            # to standard [3, num_heads, head_dim, D] format
            wqkv = _permute_wqkv(wqkv, config.num_heads, config.head_dim)
            block.Wqkv.weight = wqkv

            block.out_proj.weight = weights[f"{prefix}.inner_mha_cls.out_proj.weight"]
            block.out_proj.bias = weights[f"{prefix}.inner_mha_cls.out_proj.bias"]
            # inv_freq is recomputed by nn.RoPE, not loaded
        else:
            # Hyena block
            block.projections.weight = weights[f"{prefix}.projections.weight"]
            block.out_filter_dense.weight = weights[f"{prefix}.out_filter_dense.weight"]
            block.out_filter_dense.bias = weights[f"{prefix}.out_filter_dense.bias"]

            # Short filter
            block.short_filter.weight = weights[f"{prefix}.filter.short_filter_weight"]

            # Inner filter
            filt = block.inner_filter
            if isinstance(filt, FIRFilter):
                filt.h = weights[f"{prefix}.filter.h"]
                if filt.has_d:
                    filt.D = weights[f"{prefix}.filter.D"]
            elif isinstance(filt, IIRFilter):
                filt.log_poles = weights[f"{prefix}.filter.log_poles"]
                filt.residues = weights[f"{prefix}.filter.residues"]
                filt.D = weights[f"{prefix}.filter.D"]

    logger.info(f"Loaded weights for {config.num_layers} blocks")


def _permute_wqkv(
    wqkv: mx.array, num_heads: int, head_dim: int
) -> mx.array:
    """Permute Wqkv from per-head interleaved to standard Q,K,V layout.

    Input (column_split format): [num_heads * 3 * head_dim, hidden_size]
      arranged as [H0_Q, H0_K, H0_V, H1_Q, H1_K, H1_V, ...]
    Output (standard): [3 * num_heads * head_dim, hidden_size]
      arranged as [Q_all_heads, K_all_heads, V_all_heads]
    """
    D = wqkv.shape[1]
    # Reshape to (num_heads, 3, head_dim, D)
    w = wqkv.reshape(num_heads, 3, head_dim, D)
    # Transpose to (3, num_heads, head_dim, D)
    w = mx.transpose(w, axes=(1, 0, 2, 3))
    # Reshape back to (3*num_heads*head_dim, D)
    return w.reshape(3 * num_heads * head_dim, D)
