"""
Document ingestion for llama-memory v2.6.

Supports ingesting documents (Markdown, text, JSON, PDF) into memory with smart chunking.
Large documents are split into linked memory chunks with preserved structure.

PDF support requires the optional 'pdf' extra:
    pip install llama-memory[pdf]
"""

from __future__ import annotations

import os
import re
import json
from typing import List, Optional, Literal, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Optional PDF support
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    PDF_AVAILABLE = False


@dataclass
class Chunk:
    """A chunk of text from a document."""
    content: str
    index: int
    heading: Optional[str] = None  # For markdown, the parent heading
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)


class SmartChunker:
    """Intelligent document chunker that preserves structure.

    For Markdown: Splits on headers while preserving hierarchy
    For Text: Splits on paragraphs, then sentences if needed
    """

    def __init__(
        self,
        max_chunk_size: int = 500,
        min_chunk_size: int = 50,
        overlap: int = 50,
    ):
        """Initialize chunker.

        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters (merge smaller chunks)
            overlap: Character overlap between chunks for context
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def chunk(self, content: str, format: str = 'text') -> List[Chunk]:
        """Chunk content based on format.

        Args:
            content: Document content
            format: 'markdown', 'text', or 'json'

        Returns:
            List of Chunk objects
        """
        if format == 'markdown' or format == 'md':
            return self._chunk_markdown(content)
        elif format == 'json':
            return self._chunk_json(content)
        else:
            return self._chunk_text(content)

    def _chunk_markdown(self, content: str) -> List[Chunk]:
        """Chunk markdown by headers while preserving structure."""
        chunks = []
        lines = content.split('\n')

        current_heading = None
        current_content = []
        current_start = 0

        for i, line in enumerate(lines):
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save previous section if any
                if current_content:
                    section_text = '\n'.join(current_content).strip()
                    if section_text:
                        chunks.extend(self._split_if_too_long(
                            section_text,
                            heading=current_heading,
                            start_line=current_start
                        ))

                # Start new section
                level = len(header_match.group(1))
                current_heading = header_match.group(2)
                current_content = [line]  # Include the header
                current_start = i
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            section_text = '\n'.join(current_content).strip()
            if section_text:
                chunks.extend(self._split_if_too_long(
                    section_text,
                    heading=current_heading,
                    start_line=current_start
                ))

        # Assign sequential indexes
        for i, chunk in enumerate(chunks):
            chunk.index = i

        return chunks

    def _chunk_text(self, content: str) -> List[Chunk]:
        """Chunk plain text by paragraphs, then sentences."""
        chunks = []

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If single paragraph is too big, split by sentences
            if para_size > self.max_chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunks.append(Chunk(
                        content='\n\n'.join(current_chunk),
                        index=len(chunks)
                    ))
                    current_chunk = []
                    current_size = 0

                # Split paragraph by sentences
                sentences = self._split_sentences(para)
                sent_chunk = []
                sent_size = 0

                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue

                    if sent_size + len(sent) > self.max_chunk_size and sent_chunk:
                        chunks.append(Chunk(
                            content=' '.join(sent_chunk),
                            index=len(chunks)
                        ))
                        # Keep overlap
                        if self.overlap > 0 and sent_chunk:
                            overlap_text = ' '.join(sent_chunk[-2:])
                            if len(overlap_text) <= self.overlap:
                                sent_chunk = sent_chunk[-2:]
                            else:
                                sent_chunk = []
                        else:
                            sent_chunk = []
                        sent_size = sum(len(s) for s in sent_chunk)

                    sent_chunk.append(sent)
                    sent_size += len(sent)

                if sent_chunk:
                    chunks.append(Chunk(
                        content=' '.join(sent_chunk),
                        index=len(chunks)
                    ))
            else:
                # Check if adding this paragraph exceeds limit
                if current_size + para_size > self.max_chunk_size and current_chunk:
                    chunks.append(Chunk(
                        content='\n\n'.join(current_chunk),
                        index=len(chunks)
                    ))
                    current_chunk = []
                    current_size = 0

                current_chunk.append(para)
                current_size += para_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(Chunk(
                content='\n\n'.join(current_chunk),
                index=len(chunks)
            ))

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _chunk_json(self, content: str) -> List[Chunk]:
        """Chunk JSON by top-level keys or array items."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback to text chunking
            return self._chunk_text(content)

        chunks = []

        if isinstance(data, dict):
            for key, value in data.items():
                chunk_content = json.dumps({key: value}, indent=2)
                if len(chunk_content) > self.max_chunk_size:
                    # Split large values
                    sub_chunks = self._chunk_text(chunk_content)
                    for sc in sub_chunks:
                        sc.metadata['json_key'] = key
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(Chunk(
                        content=chunk_content,
                        index=len(chunks),
                        metadata={'json_key': key}
                    ))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                chunk_content = json.dumps(item, indent=2)
                if len(chunk_content) > self.max_chunk_size:
                    sub_chunks = self._chunk_text(chunk_content)
                    for sc in sub_chunks:
                        sc.metadata['json_index'] = i
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(Chunk(
                        content=chunk_content,
                        index=len(chunks),
                        metadata={'json_index': i}
                    ))
        else:
            # Scalar value, just return as single chunk
            chunks.append(Chunk(content=str(data), index=0))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on . ! ? followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    def _split_if_too_long(
        self,
        content: str,
        heading: Optional[str] = None,
        start_line: Optional[int] = None
    ) -> List[Chunk]:
        """Split content if it exceeds max size."""
        if len(content) <= self.max_chunk_size:
            return [Chunk(
                content=content,
                index=0,
                heading=heading,
                start_line=start_line
            )]

        # Split using text chunking
        text_chunks = self._chunk_text(content)
        for chunk in text_chunks:
            chunk.heading = heading
        return text_chunks

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks smaller than min_chunk_size."""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for i in range(1, len(chunks)):
            if len(current.content) < self.min_chunk_size:
                # Merge with next
                current = Chunk(
                    content=current.content + '\n\n' + chunks[i].content,
                    index=current.index,
                    heading=current.heading or chunks[i].heading,
                )
            else:
                merged.append(current)
                current = chunks[i]

        merged.append(current)

        # Reindex
        for i, chunk in enumerate(merged):
            chunk.index = i

        return merged


class Ingester:
    """Ingest documents into memory store.

    Handles reading, chunking, and storing documents as linked memories.
    """

    def __init__(self, store=None, chunker: Optional[SmartChunker] = None):
        """Initialize ingester.

        Args:
            store: MemoryStore instance (will get default if None)
            chunker: SmartChunker instance (will create default if None)
        """
        self.store = store
        self.chunker = chunker or SmartChunker()

    def _get_store(self):
        """Get or create memory store."""
        if self.store is None:
            from .memory import get_store
            self.store = get_store()
        return self.store

    def _detect_format(self, path: str) -> str:
        """Detect file format from extension."""
        ext = Path(path).suffix.lower()
        format_map = {
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.txt': 'text',
            '.text': 'text',
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.pdf': 'pdf',
            '.py': 'text',
            '.js': 'text',
            '.ts': 'text',
            '.yaml': 'text',
            '.yml': 'text',
            '.toml': 'text',
            '.ini': 'text',
            '.cfg': 'text',
            '.rst': 'text',
        }
        return format_map.get(ext, 'text')

    def _read_file(self, path: str) -> str:
        """Read file content."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_pdf(self, path: str) -> Tuple[str, dict]:
        """Read PDF file content with metadata.

        Args:
            path: Path to PDF file

        Returns:
            Tuple of (extracted_text, metadata_dict)

        Raises:
            ImportError: If pypdf is not installed
            ValueError: If PDF extraction fails
        """
        if not PDF_AVAILABLE:
            raise ImportError(
                "PDF support requires pypdf. Install with: pip install llama-memory[pdf]"
            )

        try:
            reader = PdfReader(path)

            # Extract metadata
            metadata = {}
            if reader.metadata:
                if reader.metadata.title:
                    metadata['title'] = str(reader.metadata.title)
                if reader.metadata.author:
                    metadata['author'] = str(reader.metadata.author)
                if reader.metadata.subject:
                    metadata['subject'] = str(reader.metadata.subject)
                if reader.metadata.creation_date:
                    metadata['created'] = str(reader.metadata.creation_date)

            metadata['page_count'] = len(reader.pages)

            # Extract text from all pages
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Clean up extracted text
                    text = text.strip()
                    # Add page marker for chunking reference
                    pages.append(f"[Page {i + 1}]\n{text}")

            full_text = '\n\n'.join(pages)

            return full_text, metadata

        except Exception as e:
            raise ValueError(f"Failed to extract PDF content: {e}")

    def ingest(
        self,
        path: str,
        type: str = 'fact',
        importance: int = 5,
        project: Optional[str] = None,
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
        format: Optional[str] = None,
        max_chunk_size: Optional[int] = None,
    ) -> List[int]:
        """Ingest a document into memory.

        Args:
            path: Path to file
            type: Memory type for chunks
            importance: Importance level for chunks
            project: Project to associate with
            context: Named context for memories
            tags: Tags to apply to all chunks
            format: Override format detection
            max_chunk_size: Override default chunk size

        Returns:
            List of created memory IDs

        Raises:
            ImportError: If PDF file and pypdf not installed
            ValueError: If file cannot be read or parsed
        """
        from .config import get_config

        store = self._get_store()
        config = get_config()

        # Detect format
        if format is None:
            format = self._detect_format(path)

        # Handle PDF files
        pdf_metadata = {}
        if format == 'pdf':
            if not config.ingestion.pdf_enabled:
                raise ValueError("PDF ingestion is disabled in config")
            content, pdf_metadata = self._read_pdf(path)
            # PDFs are processed as text after extraction
            format = 'text'
        else:
            # Read text file
            content = self._read_file(path)

        # Create chunker with custom size if specified
        chunker = self.chunker
        if max_chunk_size:
            chunker = SmartChunker(max_chunk_size=max_chunk_size)

        # Handle JSONL specially (each line is a separate record)
        if format == 'jsonl':
            return self._ingest_jsonl(
                path, content, type, importance, project, context, tags
            )

        # Chunk the document
        chunks = chunker.chunk(content, format)

        if not chunks:
            return []

        # Store chunks as linked memories
        memory_ids = []
        parent_id = None
        file_name = Path(path).name

        for chunk in chunks:
            # Build chunk metadata
            chunk_summary = self._generate_chunk_summary(chunk, file_name, pdf_metadata)
            chunk_tags = list(tags) if tags else []
            if chunk.heading:
                chunk_tags.append(f"section:{chunk.heading}")

            # Merge PDF metadata with chunk metadata
            chunk_metadata = dict(chunk.metadata)
            if pdf_metadata:
                chunk_metadata['pdf'] = pdf_metadata

            mem_id, _ = store.store(
                content=chunk.content,
                type=type,
                summary=chunk_summary,
                project=project,
                tags=chunk_tags,
                importance=importance,
                context=context,
                file_path=str(path),
                chunk_index=chunk.index,
                chunk_of=memory_ids[0] if memory_ids else None,
                source='import',
                metadata=chunk_metadata if chunk_metadata else None,
                force=True,  # Don't check duplicates for document chunks
            )

            memory_ids.append(mem_id)

        return memory_ids

    def _ingest_jsonl(
        self,
        path: str,
        content: str,
        type: str,
        importance: int,
        project: Optional[str],
        context: Optional[str],
        tags: Optional[List[str]],
    ) -> List[int]:
        """Ingest JSONL file where each line is a separate memory."""
        store = self._get_store()
        memory_ids = []

        for i, line in enumerate(content.strip().split('\n')):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                # Support either string content or object with content field
                if isinstance(data, str):
                    mem_content = data
                elif isinstance(data, dict):
                    mem_content = data.get('content', json.dumps(data))
                else:
                    mem_content = json.dumps(data)

                mem_id, _ = store.store(
                    content=mem_content,
                    type=type,
                    project=project,
                    importance=importance,
                    context=context,
                    tags=tags,
                    file_path=str(path),
                    chunk_index=i,
                    source='import',
                    force=True,
                )
                memory_ids.append(mem_id)
            except json.JSONDecodeError:
                continue

        return memory_ids

    def _generate_chunk_summary(
        self,
        chunk: Chunk,
        file_name: str,
        pdf_metadata: Optional[dict] = None
    ) -> str:
        """Generate a summary for a chunk."""
        parts = []

        # Use PDF title if available, otherwise file name
        if pdf_metadata and pdf_metadata.get('title'):
            parts.append(f"From '{pdf_metadata['title']}'")
        else:
            parts.append(f"From {file_name}")

        if chunk.heading:
            parts.append(f", section '{chunk.heading}'")

        if chunk.index > 0:
            parts.append(f" (part {chunk.index + 1})")

        # Add author if PDF
        if pdf_metadata and pdf_metadata.get('author'):
            parts.append(f" by {pdf_metadata['author']}")

        parts.append(f": {chunk.content[:50]}...")

        return ''.join(parts)

    def list_ingested(self, path: Optional[str] = None) -> List[dict]:
        """List ingested documents/files.

        Args:
            path: Optional path to filter by

        Returns:
            List of dicts with file info and chunk counts
        """
        store = self._get_store()
        conn = store.db.conn

        sql = """
            SELECT file_path, COUNT(*) as chunk_count,
                   MIN(id) as first_chunk_id,
                   MAX(importance) as max_importance,
                   MAX(created_at) as ingested_at
            FROM memories
            WHERE file_path IS NOT NULL AND archived = 0
        """
        params = []

        if path:
            sql += " AND file_path LIKE ?"
            params.append(f"%{path}%")

        sql += " GROUP BY file_path ORDER BY ingested_at DESC"

        rows = conn.execute(sql, params).fetchall()

        return [{
            'file_path': row['file_path'],
            'chunk_count': row['chunk_count'],
            'first_chunk_id': row['first_chunk_id'],
            'max_importance': row['max_importance'],
            'ingested_at': row['ingested_at'],
        } for row in rows]

    def delete_ingested(self, path: str, hard: bool = False) -> int:
        """Delete all chunks from an ingested file.

        Args:
            path: Path of ingested file
            hard: Permanently delete instead of archive

        Returns:
            Number of memories deleted
        """
        store = self._get_store()
        conn = store.db.conn

        # Get all memory IDs for this file
        rows = conn.execute("""
            SELECT id FROM memories WHERE file_path = ?
        """, (path,)).fetchall()

        count = 0
        for row in rows:
            store.delete(row['id'], hard=hard)
            count += 1

        return count


# Module-level convenience
_ingester: Optional[Ingester] = None


def get_ingester() -> Ingester:
    """Get or create ingester singleton."""
    global _ingester
    if _ingester is None:
        _ingester = Ingester()
    return _ingester


def ingest_document(path: str, **kwargs) -> List[int]:
    """Ingest a document into memory."""
    return get_ingester().ingest(path, **kwargs)


def is_pdf_available() -> bool:
    """Check if PDF support is available."""
    return PDF_AVAILABLE


def ingest_pdf(path: str, **kwargs) -> List[int]:
    """Ingest a PDF document into memory.

    Convenience wrapper that raises early if PDF support unavailable.

    Args:
        path: Path to PDF file
        **kwargs: Additional arguments passed to ingest()

    Returns:
        List of created memory IDs

    Raises:
        ImportError: If pypdf is not installed
    """
    if not PDF_AVAILABLE:
        raise ImportError(
            "PDF support requires pypdf. Install with: pip install llama-memory[pdf]"
        )
    return get_ingester().ingest(path, format='pdf', **kwargs)
