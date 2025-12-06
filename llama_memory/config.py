"""
Configuration management for llama-memory.
Handles path discovery, config loading, and platform detection.
"""

from __future__ import annotations

import os
import platform
import subprocess
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# XDG Base Directory defaults
def get_xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

def get_xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

def get_xdg_cache_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

APP_NAME = "llama-memory"
CONFIG_DIR = get_xdg_config_home() / APP_NAME
DATA_DIR = get_xdg_data_home() / APP_NAME
CACHE_DIR = get_xdg_cache_home() / APP_NAME
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Default embedding model
DEFAULT_MODEL_URL = "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2-Q4_K_M.gguf"
EMBEDDING_DIMENSIONS = 384


@dataclass
class Config:
    """Configuration for llama-memory."""

    # Database
    database_path: Path = field(default_factory=lambda: DATA_DIR / "memory.db")

    # Embedding binary (llama-embedding from llama.cpp)
    embedding_binary: Optional[Path] = None
    embedding_lib_path: Optional[Path] = None

    # Model
    embedding_model: Optional[Path] = None
    embedding_model_version: str = "all-minilm-l6-v2-Q4_K_M"
    embedding_dimensions: int = EMBEDDING_DIMENSIONS

    # SQLite-vec extension
    sqlite_vec_path: Optional[Path] = None

    # Platform info
    platform: str = field(default_factory=lambda: platform.system())
    architecture: str = field(default_factory=lambda: platform.machine())
    is_termux: bool = field(default_factory=lambda: "com.termux" in str(Path.home()))

    def __post_init__(self):
        """Auto-discover paths if not set."""
        if self.embedding_binary is None:
            self.embedding_binary = self._find_embedding_binary()
        if self.embedding_lib_path is None:
            self.embedding_lib_path = self._find_embedding_lib()
        if self.embedding_model is None:
            self.embedding_model = self._find_embedding_model()
        if self.sqlite_vec_path is None:
            self.sqlite_vec_path = self._find_sqlite_vec()

    def _find_embedding_binary(self) -> Optional[Path]:
        """Find llama-embedding binary."""
        search_paths = [
            DATA_DIR / "bin" / "llama-embedding",
            Path.home() / "Projects" / "llama.cpp" / "build" / "bin" / "llama-embedding",
            Path.home() / ".local" / "bin" / "llama-embedding",
            Path("/usr/local/bin/llama-embedding"),
        ]

        for path in search_paths:
            if path.exists() and os.access(path, os.X_OK):
                return path

        # Try which
        try:
            result = subprocess.run(["which", "llama-embedding"],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass

        return None

    def _find_embedding_lib(self) -> Optional[Path]:
        """Find llama.cpp library directory (for LD_LIBRARY_PATH)."""
        if self.embedding_binary:
            lib_dir = self.embedding_binary.parent
            if (lib_dir / "libllama.so").exists():
                return lib_dir
        return None

    def _find_embedding_model(self) -> Optional[Path]:
        """Find embedding model file."""
        search_paths = [
            DATA_DIR / "models" / DEFAULT_MODEL_NAME,
            Path.home() / "models" / DEFAULT_MODEL_NAME,
            Path.home() / ".cache" / "llama-memory" / "models" / DEFAULT_MODEL_NAME,
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _find_sqlite_vec(self) -> Optional[Path]:
        """Find sqlite-vec extension."""
        # Platform-specific extension name
        ext = ".so" if self.platform != "Darwin" else ".dylib"
        if self.platform == "Windows":
            ext = ".dll"

        search_paths = [
            DATA_DIR / "lib" / f"vec0{ext}",
            Path.home() / "lib" / f"vec0{ext}",
            Path.home() / ".local" / "lib" / f"vec0{ext}",
            Path("/usr/local/lib") / f"vec0{ext}",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        if not self.embedding_binary or not self.embedding_binary.exists():
            errors.append(f"Embedding binary not found. Run 'llama-memory install' or set embedding_binary in config.")

        if not self.embedding_model or not self.embedding_model.exists():
            errors.append(f"Embedding model not found. Run 'llama-memory install' or set embedding_model in config.")

        if not self.sqlite_vec_path or not self.sqlite_vec_path.exists():
            errors.append(f"sqlite-vec extension not found. Run 'llama-memory install' or set sqlite_vec_path in config.")

        return errors

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "database_path": str(self.database_path),
            "embedding_binary": str(self.embedding_binary) if self.embedding_binary else None,
            "embedding_lib_path": str(self.embedding_lib_path) if self.embedding_lib_path else None,
            "embedding_model": str(self.embedding_model) if self.embedding_model else None,
            "embedding_model_version": self.embedding_model_version,
            "embedding_dimensions": self.embedding_dimensions,
            "sqlite_vec_path": str(self.sqlite_vec_path) if self.sqlite_vec_path else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create from dictionary."""
        config = cls()

        if data.get("database_path"):
            config.database_path = Path(data["database_path"])
        if data.get("embedding_binary"):
            config.embedding_binary = Path(data["embedding_binary"])
        if data.get("embedding_lib_path"):
            config.embedding_lib_path = Path(data["embedding_lib_path"])
        if data.get("embedding_model"):
            config.embedding_model = Path(data["embedding_model"])
        if data.get("embedding_model_version"):
            config.embedding_model_version = data["embedding_model_version"]
        if data.get("embedding_dimensions"):
            config.embedding_dimensions = data["embedding_dimensions"]
        if data.get("sqlite_vec_path"):
            config.sqlite_vec_path = Path(data["sqlite_vec_path"])

        return config

    def save(self, path: Optional[Path] = None):
        """Save configuration to file."""
        path = path or CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from file, or create default."""
        path = path or CONFIG_FILE

        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)

        # Return default with auto-discovery
        return cls()


def get_config() -> Config:
    """Get or create configuration."""
    return Config.load()


def ensure_directories():
    """Ensure all required directories exist."""
    for dir_path in [CONFIG_DIR, DATA_DIR, CACHE_DIR,
                     DATA_DIR / "models", DATA_DIR / "lib", DATA_DIR / "bin"]:
        dir_path.mkdir(parents=True, exist_ok=True)
