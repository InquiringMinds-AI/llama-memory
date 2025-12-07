"""
Configuration management for llama-memory.
Handles path discovery, config loading, and platform detection.
"""

from __future__ import annotations

import os
import platform
import subprocess
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

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


# ============================================================================
# v2.6 Configuration Dataclasses
# ============================================================================

@dataclass
class ScoringConfig:
    """Configuration for hybrid ranking weights.

    Weights are automatically normalized to sum to 1.0.
    """
    semantic: float = 0.40      # Vector similarity weight
    importance: float = 0.25    # User-assigned importance weight
    recency: float = 0.15       # Time decay weight
    frequency: float = 0.10     # Access count weight
    confidence: float = 0.05    # Memory certainty weight
    entity_match: float = 0.05  # Entity overlap bonus weight

    def __post_init__(self):
        """Validate and normalize weights to sum to 1.0."""
        # Clamp negative values to 0
        self.semantic = max(0.0, float(self.semantic))
        self.importance = max(0.0, float(self.importance))
        self.recency = max(0.0, float(self.recency))
        self.frequency = max(0.0, float(self.frequency))
        self.confidence = max(0.0, float(self.confidence))
        self.entity_match = max(0.0, float(self.entity_match))

        # Normalize to sum to 1.0
        total = (self.semantic + self.importance + self.recency +
                 self.frequency + self.confidence + self.entity_match)

        if total > 0 and abs(total - 1.0) > 0.001:
            self.semantic /= total
            self.importance /= total
            self.recency /= total
            self.frequency /= total
            self.confidence /= total
            self.entity_match /= total

    def to_dict(self) -> Dict[str, float]:
        return {
            'semantic': self.semantic,
            'importance': self.importance,
            'recency': self.recency,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'entity_match': self.entity_match,
        }


@dataclass
class EntityConfig:
    """Configuration for entity extraction."""
    enabled: bool = True
    extract_names: bool = True  # Enable heuristic name extraction
    max_entities: int = 20      # Cap entities per memory
    types: List[str] = field(default_factory=lambda: [
        "person", "project", "tool", "concept", "organization"
    ])
    additional_tools: List[str] = field(default_factory=list)
    additional_orgs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'extract_names': self.extract_names,
            'max_entities': self.max_entities,
            'types': self.types,
            'additional_tools': self.additional_tools,
            'additional_orgs': self.additional_orgs,
        }


@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""
    pdf_enabled: bool = True
    max_chunk_size: int = 500
    overlap: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pdf_enabled': self.pdf_enabled,
            'max_chunk_size': self.max_chunk_size,
            'overlap': self.overlap,
        }


@dataclass
class BudgetConfig:
    """Configuration for token budget management."""
    enabled: bool = True
    max_tokens: int = 4000
    summary_tokens: int = 25
    overhead_tokens: int = 100
    content_allocation: float = 0.70

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'max_tokens': self.max_tokens,
            'summary_tokens': self.summary_tokens,
            'overhead_tokens': self.overhead_tokens,
            'content_allocation': self.content_allocation,
        }


@dataclass
class DecayConfig:
    """Configuration for memory decay."""
    enabled: bool = True
    start_days: int = 120       # Days until decay starts affecting ranking
    archive_days: int = 180     # Days until auto-archive
    min_importance: int = 3     # Minimum importance to decay

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'start_days': self.start_days,
            'archive_days': self.archive_days,
            'min_importance': self.min_importance,
        }


# ============================================================================
# Main Config Class
# ============================================================================

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

    # v2.6 configuration sections
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    entities: EntityConfig = field(default_factory=EntityConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)

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
            # v2.6 sections
            "scoring": self.scoring.to_dict(),
            "entities": self.entities.to_dict(),
            "ingestion": self.ingestion.to_dict(),
            "budget": self.budget.to_dict(),
            "decay": self.decay.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create from dictionary, ignoring unknown keys."""
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

        # v2.6 sections
        if "scoring" in data and isinstance(data["scoring"], dict):
            sc = data["scoring"]
            config.scoring = ScoringConfig(
                semantic=float(sc.get("semantic", 0.40)),
                importance=float(sc.get("importance", 0.25)),
                recency=float(sc.get("recency", 0.15)),
                frequency=float(sc.get("frequency", 0.10)),
                confidence=float(sc.get("confidence", 0.05)),
                entity_match=float(sc.get("entity_match", 0.05)),
            )

        if "entities" in data and isinstance(data["entities"], dict):
            ent = data["entities"]
            config.entities = EntityConfig(
                enabled=bool(ent.get("enabled", True)),
                extract_names=bool(ent.get("extract_names", True)),
                max_entities=int(ent.get("max_entities", 20)),
                types=ent.get("types", config.entities.types),
                additional_tools=ent.get("additional_tools", []),
                additional_orgs=ent.get("additional_orgs", []),
            )

        if "ingestion" in data and isinstance(data["ingestion"], dict):
            ing = data["ingestion"]
            config.ingestion = IngestionConfig(
                pdf_enabled=bool(ing.get("pdf_enabled", True)),
                max_chunk_size=int(ing.get("max_chunk_size", 500)),
                overlap=int(ing.get("overlap", 50)),
            )

        if "budget" in data and isinstance(data["budget"], dict):
            bud = data["budget"]
            config.budget = BudgetConfig(
                enabled=bool(bud.get("enabled", True)),
                max_tokens=int(bud.get("max_tokens", 4000)),
                summary_tokens=int(bud.get("summary_tokens", 25)),
                overhead_tokens=int(bud.get("overhead_tokens", 100)),
                content_allocation=float(bud.get("content_allocation", 0.70)),
            )

        if "decay" in data and isinstance(data["decay"], dict):
            dec = data["decay"]
            config.decay = DecayConfig(
                enabled=bool(dec.get("enabled", True)),
                start_days=int(dec.get("start_days", 120)),
                archive_days=int(dec.get("archive_days", 180)),
                min_importance=int(dec.get("min_importance", 3)),
            )

        return config

    def save(self, path: Optional[Path] = None):
        """Save configuration to file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required to save config. Install with: pip install pyyaml")

        path = path or CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from file, or create default.

        Gracefully handles:
        - Missing config file (returns defaults)
        - Missing PyYAML (returns defaults with warning)
        - Malformed YAML (returns defaults with warning)
        """
        path = path or CONFIG_FILE

        if not path.exists():
            return cls()

        if not YAML_AVAILABLE:
            warnings.warn("PyYAML not installed, using default config")
            return cls()

        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        except Exception as e:
            warnings.warn(f"Error loading config from {path}: {e}. Using defaults.")
            return cls()


# Global config instance (lazy loaded and cached)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance (lazy loaded and cached)."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reload_config(path: Optional[Path] = None) -> Config:
    """Reload config from file, updating the global instance."""
    global _config
    _config = Config.load(path)
    return _config


def set_config(config: Config):
    """Set the global config instance (primarily for testing)."""
    global _config
    _config = config


def ensure_directories():
    """Ensure all required directories exist."""
    for dir_path in [CONFIG_DIR, DATA_DIR, CACHE_DIR,
                     DATA_DIR / "models", DATA_DIR / "lib", DATA_DIR / "bin"]:
        dir_path.mkdir(parents=True, exist_ok=True)
