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
NIM_FORWARD_ENDPOINT = "/biology/arc/evo2/forward"
NIM_GENERATE_ENDPOINT = "/biology/arc/evo2/generate"

# Evo2 model specifications
EVO2_MODELS = {
    "evo2_1b_base": {
        "size_gb": 2.5,
        "max_context": 8192,
        "embed_dim": 2048,
        "recommended_layer": "blocks.14.mlp.l3",
    },
    "evo2_7b": {
        "size_gb": 16,
        "max_context": 1_000_000,
        "embed_dim": 4096,
        "recommended_layer": "blocks.28.mlp.l3",
    },
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
