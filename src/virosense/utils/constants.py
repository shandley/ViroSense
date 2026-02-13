"""Constants and configuration for virosense."""

import os
from pathlib import Path

# Environment variable names
ENV_VIROSENSE_DATA = "VIROSENSE_DATA_DIR"
ENV_VIROSENSE_CACHE = "VIROSENSE_CACHE_DIR"
ENV_NVIDIA_API_KEY = "NVIDIA_API_KEY"

# Default directories
DEFAULT_DATA_DIR = Path.home() / ".virosense"
DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR / "cache"
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "models"

# NVIDIA NIM endpoints
NIM_BASE_URL = "https://health.api.nvidia.com/v1"
NIM_FORWARD_ENDPOINT = "/biology/arc/evo2-40b/forward"
NIM_GENERATE_ENDPOINT = "/biology/arc/evo2-40b/generate"

# NIM API constraints
NIM_MAX_SEQUENCE_LENGTH = 16_000  # bp per request
NIM_RATE_LIMIT_RPM = 40
NIM_REQUEST_DELAY = 60.0 / NIM_RATE_LIMIT_RPM  # seconds between requests
NIM_REQUEST_TIMEOUT = 120.0  # seconds

# Evo2 model specifications
EVO2_MODELS = {
    "evo2_1b_base": {
        "size_gb": 2.5,
        "max_context": 8192,
        "embed_dim": 2048,
        "num_layers": 25,
        "recommended_layer": "blocks.14.mlp.l3",
    },
    "evo2_7b": {
        "size_gb": 16,
        "max_context": 1_000_000,
        "embed_dim": 4096,
        "num_layers": 32,
        "recommended_layer": "blocks.28.mlp.l3",
    },
    "evo2_40b": {
        "size_gb": 80,
        "max_context": 1_000_000,
        "embed_dim": 4096,
        "num_layers": 50,
        "recommended_layer": "blocks.20.mlp.l3",
        "nim_endpoint": NIM_FORWARD_ENDPOINT,
    },
}

# Attention layer indices (TransformerLayers, not HyenaLayers)
EVO2_ATTENTION_LAYERS = {
    "evo2_7b": [3, 10, 17, 24, 31],
    "evo2_40b": [3, 10, 17, 24, 31, 35, 42, 49],
}

# Default settings
DEFAULT_EVO2_LAYER = "blocks.28.mlp.l3"
DEFAULT_EVO2_MODEL = "evo2_7b"
DEFAULT_BACKEND = "nim"

# ProstT5 (via vHold)
PROSTT5_EMBED_DIM = 1024

# Detection
DEFAULT_VIRAL_THRESHOLD = 0.5
DEFAULT_MIN_CONTIG_LENGTH = 500

# Clustering
DEFAULT_CLUSTERING_ALGORITHM = "hdbscan"
DEFAULT_MIN_CLUSTER_SIZE = 5

# Classification
DEFAULT_CLASSIFIER_HIDDEN_DIMS = [512, 128]
DEFAULT_CLASSIFIER_EPOCHS = 50
DEFAULT_CLASSIFIER_LR = 1e-3


def get_data_dir() -> Path:
    """Get the virosense data directory."""
    env_path = os.environ.get(ENV_VIROSENSE_DATA)
    if env_path:
        return Path(env_path)
    return DEFAULT_DATA_DIR


def get_cache_dir() -> Path:
    """Get the embedding cache directory."""
    env_path = os.environ.get(ENV_VIROSENSE_CACHE)
    if env_path:
        return Path(env_path)
    return get_data_dir() / "cache"


def get_nvidia_api_key() -> str | None:
    """Get NVIDIA API key from environment."""
    return os.environ.get(ENV_NVIDIA_API_KEY)


# Layer name translation: native Evo2 (blocks.*) <-> NIM API (decoder.layers.*)
# Native: blocks.N.mlp.l3       -> NIM: decoder.layers.N.mlp.linear_fc2
# Native: blocks.N.mlp.l1/l2    -> NIM: decoder.layers.N.mlp.linear_fc1 (fused gate+up)
# Native: blocks.N.mixer        -> NIM: decoder.layers.N.mixer
# Native: blocks.N.self_attention -> NIM: decoder.layers.N.self_attention

_NATIVE_TO_NIM_SUBLAYER = {
    ".mlp.l3": ".mlp.linear_fc2",
    ".mlp.l1": ".mlp.linear_fc1",
    ".mlp.l2": ".mlp.linear_fc1",
}

# Special layer names that are the same in both conventions
_NIM_SPECIAL_LAYERS = {"embedding", "decoder.final_norm", "output_layer"}


def translate_layer_to_nim(native_name: str) -> str:
    """Translate native Evo2 layer name to NIM API layer name.

    Args:
        native_name: Layer name in native format (e.g. 'blocks.28.mlp.l3')
            or already in NIM format.

    Returns:
        Layer name in NIM API format (e.g. 'decoder.layers.28.mlp.linear_fc2').
    """
    if native_name in _NIM_SPECIAL_LAYERS:
        return native_name
    if native_name.startswith("decoder.layers."):
        return native_name

    if not native_name.startswith("blocks."):
        raise ValueError(
            f"Unknown layer name format: {native_name!r}. "
            "Expected 'blocks.[n].*' (native) or 'decoder.layers.[n].*' (NIM)."
        )

    nim_name = native_name.replace("blocks.", "decoder.layers.", 1)
    for native_suffix, nim_suffix in _NATIVE_TO_NIM_SUBLAYER.items():
        if nim_name.endswith(native_suffix):
            nim_name = nim_name[: -len(native_suffix)] + nim_suffix
            break
    return nim_name


def translate_layer_to_native(nim_name: str) -> str:
    """Translate NIM API layer name back to native Evo2 format.

    Args:
        nim_name: Layer name in NIM format (e.g. 'decoder.layers.28.mlp.linear_fc2').

    Returns:
        Layer name in native format (e.g. 'blocks.28.mlp.l3').
    """
    if nim_name in _NIM_SPECIAL_LAYERS:
        return nim_name
    if nim_name.startswith("blocks."):
        return nim_name

    if not nim_name.startswith("decoder.layers."):
        raise ValueError(
            f"Unknown layer name format: {nim_name!r}. "
            "Expected 'decoder.layers.[n].*' (NIM) or 'blocks.[n].*' (native)."
        )

    native_name = nim_name.replace("decoder.layers.", "blocks.", 1)
    native_name = native_name.replace(".mlp.linear_fc2", ".mlp.l3")
    native_name = native_name.replace(".mlp.linear_fc1", ".mlp.l1")
    return native_name
