"""
Token budget management for llama-memory.

Implements the dual-response pattern:
- Returns ALL memory summaries (cheap, complete index)
- Returns top N full memories within token budget
- Provides metadata about what was included/excluded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import Memory

from .summarizer import estimate_tokens


@dataclass
class MemorySummary:
    """Summary representation of a memory for the index."""
    id: int
    summary: str
    importance: int
    type: str
    distance: Optional[float] = None
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'summary': self.summary,
            'importance': self.importance,
            'type': self.type,
            'distance': self.distance,
            'score': self.score,
        }


@dataclass
class RecallResponse:
    """Response from token-budgeted recall.

    Contains:
    - summaries: All matching memories as summaries (cheap index)
    - full_memories: Top memories with full content (within budget)
    - Metadata about the recall operation
    """
    summaries: List[MemorySummary]
    full_memories: List['Memory']
    query: Optional[str] = None
    total_matches: int = 0
    tokens_used: int = 0
    tokens_budget: int = 4000
    summary_tokens: int = 0
    content_tokens: int = 0
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            'query': self.query,
            'total_matches': self.total_matches,
            'tokens_used': self.tokens_used,
            'tokens_budget': self.tokens_budget,
            'summary_tokens': self.summary_tokens,
            'content_tokens': self.content_tokens,
            'truncated': self.truncated,
            'summaries': [s.to_dict() for s in self.summaries],
            'full_memories': [m.to_dict() for m in self.full_memories],
        }


class TokenBudgetManager:
    """Manages token allocation for memory recall.

    Implements the dual-response pattern from claude-memory-mcp:
    1. Always include ALL summaries (~20-25 tokens each)
    2. Fill remaining budget with full memory content
    3. Prioritize by relevance score

    This allows Claude to know what memories exist without
    exploding context, while still getting full content for
    the most relevant memories.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        summary_tokens: int = 25,
        overhead_tokens: int = 100,
        content_allocation: float = 0.70,
    ):
        """Initialize token budget manager.

        Args:
            max_tokens: Maximum total tokens for recall response
            summary_tokens: Estimated tokens per summary
            overhead_tokens: Tokens for formatting and metadata
            content_allocation: Fraction of remaining budget for content (0-1)
        """
        self.max_tokens = max_tokens
        self.summary_tokens = summary_tokens
        self.overhead_tokens = overhead_tokens
        self.content_allocation = content_allocation

    def allocate(
        self,
        memories: List['Memory'],
        query: Optional[str] = None,
    ) -> RecallResponse:
        """Allocate memories within token budget.

        Args:
            memories: List of memories sorted by relevance
            query: Optional query string for metadata

        Returns:
            RecallResponse with summaries and full memories
        """
        # Handle zero or negative budget
        if self.max_tokens <= 0:
            return RecallResponse(
                summaries=[],
                full_memories=[],
                query=query,
                total_matches=len(memories) if memories else 0,
                tokens_used=0,
                tokens_budget=0,
            )

        if not memories:
            return RecallResponse(
                summaries=[],
                full_memories=[],
                query=query,
                total_matches=0,
                tokens_used=0,
                tokens_budget=self.max_tokens,
            )

        # Step 1: Always include ALL summaries
        summaries = []
        total_summary_tokens = 0

        for mem in memories:
            summary_text = mem.summary or self._generate_fallback_summary(mem.content)
            summary_tokens = estimate_tokens(summary_text)
            total_summary_tokens += summary_tokens

            summaries.append(MemorySummary(
                id=mem.id,
                summary=summary_text,
                importance=mem.importance,
                type=mem.type,
                distance=mem.distance,
                score=mem.score,
            ))

        # Step 2: Calculate content budget
        available = self.max_tokens - self.overhead_tokens - total_summary_tokens
        content_budget = int(available * self.content_allocation)

        # Ensure minimum content budget
        if content_budget < 100:
            content_budget = min(100, available)

        # Step 3: Fill with full memories within budget
        full_memories = []
        total_content_tokens = 0

        for mem in memories:
            # Use stored token count or estimate
            mem_tokens = getattr(mem, 'token_count', None) or estimate_tokens(mem.content)

            if total_content_tokens + mem_tokens <= content_budget:
                full_memories.append(mem)
                total_content_tokens += mem_tokens
            elif total_content_tokens == 0:
                # Always include at least one memory if possible
                full_memories.append(mem)
                total_content_tokens += mem_tokens
                break

        # Step 4: Build response
        total_tokens = self.overhead_tokens + total_summary_tokens + total_content_tokens

        return RecallResponse(
            summaries=summaries,
            full_memories=full_memories,
            query=query,
            total_matches=len(memories),
            tokens_used=total_tokens,
            tokens_budget=self.max_tokens,
            summary_tokens=total_summary_tokens,
            content_tokens=total_content_tokens,
            truncated=len(full_memories) < len(memories),
        )

    def _generate_fallback_summary(self, content: str, max_words: int = 20) -> str:
        """Generate a fallback summary if none exists.

        Uses simple truncation for speed (proper summarization
        should happen at store time).
        """
        if not content:
            return ""
        words = content.split()[:max_words]
        result = ' '.join(words)
        if len(content.split()) > max_words:
            result += '...'
        return result

    def estimate_recall_cost(
        self,
        memory_count: int,
        avg_content_length: int = 500,
    ) -> dict:
        """Estimate token costs for a recall operation.

        Useful for planning and optimization.

        Args:
            memory_count: Number of memories to recall
            avg_content_length: Average content length in characters

        Returns:
            Dict with cost estimates
        """
        summary_cost = memory_count * self.summary_tokens
        avg_content_tokens = avg_content_length // 4 + 10
        max_full_memories = max(1, (self.max_tokens - self.overhead_tokens - summary_cost) // avg_content_tokens)

        return {
            'memory_count': memory_count,
            'summary_tokens': summary_cost,
            'max_full_memories': max_full_memories,
            'estimated_content_tokens': min(max_full_memories, memory_count) * avg_content_tokens,
            'total_budget': self.max_tokens,
            'will_truncate': memory_count > max_full_memories,
        }


# Default budget manager instance
_default_manager: Optional[TokenBudgetManager] = None


def get_budget_manager(max_tokens: int = 4000) -> TokenBudgetManager:
    """Get or create default budget manager."""
    global _default_manager
    if _default_manager is None or _default_manager.max_tokens != max_tokens:
        _default_manager = TokenBudgetManager(max_tokens=max_tokens)
    return _default_manager


def budgeted_recall(
    memories: List['Memory'],
    max_tokens: int = 4000,
    query: Optional[str] = None,
) -> RecallResponse:
    """Convenience function for token-budgeted recall.

    Args:
        memories: Memories sorted by relevance
        max_tokens: Token budget
        query: Optional query for metadata

    Returns:
        RecallResponse with summaries and full memories
    """
    manager = get_budget_manager(max_tokens)
    return manager.allocate(memories, query)
