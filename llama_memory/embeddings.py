"""
Embedding generation using llama.cpp with caching.
"""

from __future__ import annotations

import os
import hashlib
import subprocess
from typing import Optional
from collections import OrderedDict
from .config import Config, get_config


class EmbeddingError(Exception):
    """Error generating embeddings."""
    pass


class LRUCache:
    """Simple LRU cache for embeddings."""

    def __init__(self, maxsize: int = 1000):
        self.cache: OrderedDict[str, list[float]] = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        key = self._hash(text)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, text: str, embedding: list[float]):
        key = self._hash(text)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = embedding

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{(self.hits / total * 100):.1f}%" if total > 0 else "0%"
        }


class EmbeddingGenerator:
    """Generate embeddings using llama.cpp with LRU cache."""

    def __init__(self, config: Optional[Config] = None, cache_size: int = 1000):
        self.config = config or get_config()
        self.cache = LRUCache(cache_size)
        self._validate()

    def _validate(self):
        """Validate embedding dependencies are available."""
        if not self.config.embedding_binary or not self.config.embedding_binary.exists():
            raise EmbeddingError(
                f"Embedding binary not found at {self.config.embedding_binary}. "
                "Run 'llama-memory install' to set up dependencies."
            )

        if not self.config.embedding_model or not self.config.embedding_model.exists():
            raise EmbeddingError(
                f"Embedding model not found at {self.config.embedding_model}. "
                "Run 'llama-memory install' to download the model."
            )

    def _get_env(self) -> dict:
        """Get environment with library path set."""
        env = os.environ.copy()
        if self.config.embedding_lib_path:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{self.config.embedding_lib_path}:{existing}"
        return env

    def generate(self, text: str, use_cache: bool = True) -> list[float]:
        """Generate embedding for text, using cache if available."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")

        # Check cache first
        if use_cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        # Sanitize text: replace newlines with spaces to avoid breaking llama-embedding output
        sanitized_text = ' '.join(text.split())

        try:
            result = subprocess.run(
                [
                    str(self.config.embedding_binary),
                    "-m", str(self.config.embedding_model),
                    "-p", sanitized_text,
                    "--embd-normalize", "2",  # L2 normalization
                ],
                capture_output=True,
                text=True,
                env=self._get_env(),
                timeout=60,
            )

            if result.returncode != 0:
                raise EmbeddingError(f"Embedding generation failed: {result.stderr}")

            # Parse embedding from output (check both stdout and stderr)
            all_output = result.stdout + "\n" + result.stderr
            for line in all_output.split("\n"):
                if line.startswith("embedding 0:"):
                    values = line.replace("embedding 0:", "").strip().split()
                    embedding = [float(v) for v in values]

                    if len(embedding) != self.config.embedding_dimensions:
                        raise EmbeddingError(
                            f"Expected {self.config.embedding_dimensions} dimensions, "
                            f"got {len(embedding)}"
                        )

                    # Store in cache
                    if use_cache:
                        self.cache.put(text, embedding)

                    return embedding

            raise EmbeddingError(f"No embedding found in output: {all_output[:500]}")

        except subprocess.TimeoutExpired:
            raise EmbeddingError("Embedding generation timed out")
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"Embedding generation error: {e}")

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.stats()

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # For now, just iterate. Could optimize with batching later.
        return [self.generate(text) for text in texts]


# Module-level convenience function
_generator: Optional[EmbeddingGenerator] = None


def get_embedding(text: str, config: Optional[Config] = None) -> list[float]:
    """Get embedding for text (uses cached generator)."""
    global _generator
    if _generator is None or (config and config != _generator.config):
        _generator = EmbeddingGenerator(config)
    return _generator.generate(text)
