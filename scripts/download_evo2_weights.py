#!/usr/bin/env python3
"""Download Evo2 7B weights from HuggingFace for MLX backend.

Downloads the safetensors checkpoint from ishanjmukherjee/evo2-7b
to ~/.virosense/models/evo2-7b/ (or a custom directory).

Usage:
    python scripts/download_evo2_weights.py
    python scripts/download_evo2_weights.py --output-dir /path/to/models/evo2-7b
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "ishanjmukherjee/evo2-7b"
FILENAME = "model.safetensors"


def download_weights(output_dir: Path) -> Path:
    """Download Evo2 7B safetensors weights.

    Args:
        output_dir: Directory to save the weights.

    Returns:
        Path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / FILENAME

    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"Weights already exist at {output_path} ({size_gb:.1f} GB)")
        return output_path

    print(f"Downloading {REPO_ID}/{FILENAME} (~13.2 GB)...")
    print(f"Destination: {output_dir}")

    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(output_dir),
    )

    print(f"Download complete: {downloaded}")
    return Path(downloaded)


def main() -> None:
    from virosense.utils.constants import MLX_MODEL_DIR

    parser = argparse.ArgumentParser(description="Download Evo2 7B weights for MLX backend")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MLX_MODEL_DIR,
        help=f"Output directory (default: {MLX_MODEL_DIR})",
    )
    args = parser.parse_args()

    download_weights(args.output_dir)


if __name__ == "__main__":
    main()
