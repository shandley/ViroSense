"""Configuration dataclasses for virosense pipelines."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Evo2Config:
    """Configuration for Evo2 inference."""
    backend: str = "nim"
    model: str = "evo2_7b"
    layer: str = "blocks.28.mlp.l3"
    batch_size: int = 16
    cache_dir: Path | None = None
    api_key: str | None = None
    device: str = "cuda:0"


@dataclass
class ProstT5Config:
    """Configuration for ProstT5 (via vHold)."""
    device: str = "cpu"
    model_dir: Path | None = None
    batch_size: int = 32
    fast: bool = True


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    evo2: Evo2Config = field(default_factory=Evo2Config)
    prostt5: ProstT5Config = field(default_factory=ProstT5Config)
    threads: int = 4
    output_dir: Path = field(default_factory=lambda: Path("output"))
