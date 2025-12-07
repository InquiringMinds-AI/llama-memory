"""
Auto-summarization for llama-memory.

Provides multiple summarization strategies:
- HeuristicSummarizer: Fast, no dependencies (default)
- ExtractiveSummarizer: Uses embeddings for extractive summarization
- LLMSummarizer: High quality, requires local LLM (optional)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional, List

# Stop words for heuristic summarization (common English words)
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
}


class Summarizer(ABC):
    """Abstract base class for summarizers."""

    @abstractmethod
    def summarize(self, content: str, max_words: int = 20) -> str:
        """Generate a summary of the content.

        Args:
            content: Text to summarize
            max_words: Maximum number of words in summary

        Returns:
            Summary string
        """
        pass


class HeuristicSummarizer(Summarizer):
    """Fast heuristic summarizer with no external dependencies.

    Uses multiple strategies:
    1. First sentence if short enough
    2. Key phrase extraction (remove stop words)
    3. Truncation with ellipsis
    """

    def summarize(self, content: str, max_words: int = 20) -> str:
        if not content or not content.strip():
            return ""

        # Clean and normalize whitespace
        text = ' '.join(content.split())

        # Strategy 1: Try to use first sentence
        first_sentence = self._get_first_sentence(text)
        if first_sentence:
            words = first_sentence.split()
            if len(words) <= max_words:
                return first_sentence

        # Strategy 2: Extract key phrases from first sentence
        if first_sentence:
            key_summary = self._extract_key_phrases(first_sentence, max_words)
            if key_summary:
                return key_summary

        # Strategy 3: Fall back to first N content words
        return self._truncate_to_words(text, max_words)

    def _get_first_sentence(self, text: str) -> Optional[str]:
        """Extract the first sentence from text."""
        # Split on sentence-ending punctuation
        match = re.match(r'^([^.!?]+[.!?])', text)
        if match:
            sentence = match.group(1).strip()
            # Avoid very short sentences (likely abbreviations)
            if len(sentence.split()) >= 3:
                return sentence

        # No clear sentence boundary, try first line
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Remove markdown headers
            first_line = re.sub(r'^#+\s*', '', first_line)
            if first_line and len(first_line.split()) >= 3:
                return first_line

        return None

    def _extract_key_phrases(self, text: str, max_words: int) -> Optional[str]:
        """Extract key phrases by removing stop words."""
        words = text.split()

        # Keep words that are not stop words and are meaningful
        key_words = []
        for word in words:
            # Clean punctuation for comparison
            clean_word = re.sub(r'[^\w\s-]', '', word.lower())
            if clean_word and clean_word not in STOP_WORDS:
                key_words.append(word)
            elif len(key_words) > 0 and len(key_words) < max_words:
                # Keep some structure by including connecting words
                if word.lower() in ('and', 'or', 'for', 'with', 'to'):
                    key_words.append(word)

        if len(key_words) <= max_words:
            result = ' '.join(key_words)
            # Clean up trailing punctuation and add ellipsis if truncated
            result = re.sub(r'[,;:]+$', '', result)
            return result if result else None

        # Truncate key words
        result = ' '.join(key_words[:max_words])
        result = re.sub(r'[,;:]+$', '', result)
        return result + '...' if key_words else None

    def _truncate_to_words(self, text: str, max_words: int) -> str:
        """Simple truncation to max words."""
        words = text.split()[:max_words]
        result = ' '.join(words)
        # Clean trailing punctuation
        result = re.sub(r'[,;:]+$', '', result)
        if len(text.split()) > max_words:
            result += '...'
        return result


class ExtractiveSummarizer(Summarizer):
    """Extractive summarizer using embeddings to find most representative sentence.

    Finds the sentence most similar to the document centroid.
    Requires embedding capability.
    """

    def __init__(self, embed_func=None, config=None):
        """Initialize with embedding function.

        Args:
            embed_func: Function that takes text and returns embedding vector
            config: Optional config for embedding
        """
        self.embed_func = embed_func
        self.config = config

    def summarize(self, content: str, max_words: int = 20) -> str:
        if not content or not content.strip():
            return ""

        sentences = self._split_sentences(content)
        if len(sentences) == 0:
            return ""
        if len(sentences) == 1:
            return self._truncate(sentences[0], max_words)

        # Get embeddings for each sentence
        if not self.embed_func:
            # Fall back to heuristic if no embedding function
            return HeuristicSummarizer().summarize(content, max_words)

        try:
            embeddings = [self.embed_func(s) for s in sentences]

            # Compute centroid
            dim = len(embeddings[0])
            centroid = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]

            # Find sentence closest to centroid
            best_idx = 0
            best_sim = -1
            for i, emb in enumerate(embeddings):
                sim = self._cosine_similarity(emb, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            return self._truncate(sentences[best_idx], max_words)

        except Exception:
            # Fall back to heuristic on any error
            return HeuristicSummarizer().summarize(content, max_words)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

    def _truncate(self, text: str, max_words: int) -> str:
        """Truncate to max words."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words]) + '...'

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class LLMSummarizer(Summarizer):
    """LLM-based summarizer for highest quality summaries.

    Uses a local LLM via Ollama or llama.cpp for summarization.
    Falls back to heuristic if LLM is unavailable.
    """

    def __init__(self, model: str = "llama3.2:1b", provider: str = "ollama"):
        """Initialize LLM summarizer.

        Args:
            model: Model name to use
            provider: "ollama" or "llamacpp"
        """
        self.model = model
        self.provider = provider
        self._available = None

    def is_available(self) -> bool:
        """Check if LLM is available."""
        if self._available is not None:
            return self._available

        if self.provider == "ollama":
            try:
                import subprocess
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    timeout=5
                )
                self._available = result.returncode == 0
            except Exception:
                self._available = False
        else:
            self._available = False

        return self._available

    def summarize(self, content: str, max_words: int = 20) -> str:
        if not content or not content.strip():
            return ""

        if not self.is_available():
            return HeuristicSummarizer().summarize(content, max_words)

        try:
            import subprocess

            prompt = f"Summarize the following in exactly {max_words} words or fewer. Output ONLY the summary, no explanation:\n\n{content[:1000]}"

            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                summary = result.stdout.strip()
                # Truncate if LLM exceeded word limit
                words = summary.split()
                if len(words) > max_words:
                    summary = ' '.join(words[:max_words]) + '...'
                return summary

        except Exception:
            pass

        # Fall back to heuristic
        return HeuristicSummarizer().summarize(content, max_words)


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a conservative estimate of ~4 characters per token.
    This is approximate but works for most English text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Average ~4 chars per token, add small overhead
    return len(text) // 4 + 10


def get_summarizer(provider: str = "heuristic", **kwargs) -> Summarizer:
    """Get summarizer instance by provider name.

    Args:
        provider: "heuristic", "extractive", or "llm"
        **kwargs: Additional arguments for the summarizer

    Returns:
        Summarizer instance
    """
    if provider == "extractive":
        return ExtractiveSummarizer(**kwargs)
    elif provider == "llm":
        return LLMSummarizer(**kwargs)
    else:
        return HeuristicSummarizer()


# Default summarizer instance
_default_summarizer: Optional[Summarizer] = None


def get_default_summarizer() -> Summarizer:
    """Get the default summarizer (heuristic)."""
    global _default_summarizer
    if _default_summarizer is None:
        _default_summarizer = HeuristicSummarizer()
    return _default_summarizer


def auto_summarize(content: str, max_words: int = 20) -> str:
    """Convenience function to auto-summarize content.

    Args:
        content: Text to summarize
        max_words: Maximum words in summary

    Returns:
        Summary string
    """
    return get_default_summarizer().summarize(content, max_words)
