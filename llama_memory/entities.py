"""
Entity extraction for llama-memory v2.5.

Extracts named entities from memory content for enhanced search and organization.
Supports: projects, tools, people, concepts, organizations, locations.
"""

from __future__ import annotations

import re
import json
from typing import List, Optional, Literal, Dict, Set
from dataclasses import dataclass
from datetime import datetime


EntityType = Literal['person', 'project', 'tool', 'concept', 'organization', 'location']


@dataclass
class Entity:
    """An extracted entity."""
    name: str
    normalized: str
    type: EntityType

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'normalized': self.normalized,
            'type': self.type,
        }


class EntityExtractor:
    """Extract named entities from text content.

    Extracts:
    - Projects: kebab-case identifiers (llama-memory, claude-code)
    - Tools: Known developer tools, languages, frameworks
    - People: Capitalized names, proper nouns
    - Concepts: Technical terms
    - Organizations: Company/org names
    """

    # Known development tools, languages, and frameworks
    KNOWN_TOOLS: Set[str] = {
        # Languages
        'python', 'javascript', 'typescript', 'rust', 'go', 'golang', 'java',
        'c', 'cpp', 'c++', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'perl',
        'bash', 'shell', 'zsh', 'lua', 'r', 'julia', 'elixir', 'clojure',
        'haskell', 'ocaml', 'fsharp', 'dart', 'zig', 'nim', 'crystal',

        # Databases
        'sqlite', 'postgresql', 'postgres', 'mysql', 'mariadb', 'mongodb',
        'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'firestore',
        'sqlite-vec', 'chromadb', 'pinecone', 'weaviate', 'qdrant', 'milvus',

        # Frameworks/Libraries
        'react', 'vue', 'angular', 'svelte', 'nextjs', 'nuxt', 'gatsby',
        'django', 'flask', 'fastapi', 'express', 'nestjs', 'rails',
        'spring', 'laravel', 'phoenix', 'actix', 'axum', 'rocket',
        'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'pandas', 'numpy',

        # Build/Package tools
        'git', 'docker', 'kubernetes', 'k8s', 'npm', 'yarn', 'pnpm',
        'pip', 'poetry', 'conda', 'cargo', 'gradle', 'maven', 'brew',
        'homebrew', 'apt', 'pacman', 'yum', 'dnf',

        # Dev tools
        'vscode', 'vim', 'neovim', 'nvim', 'emacs', 'jetbrains', 'intellij',
        'cursor', 'windsurf', 'zed', 'sublime',

        # AI/ML tools
        'ollama', 'llama', 'llama.cpp', 'llamacpp', 'gguf', 'ggml',
        'transformers', 'langchain', 'llamaindex', 'openai', 'anthropic',
        'claude', 'gpt', 'gemini', 'mistral', 'huggingface',

        # Protocols/Standards
        'mcp', 'json-rpc', 'jsonrpc', 'rest', 'graphql', 'grpc', 'websocket',
        'http', 'https', 'ssh', 'ftp', 'smtp',

        # Mobile/Platform
        'android', 'ios', 'linux', 'macos', 'windows', 'termux',
        'react-native', 'flutter', 'expo',

        # Services
        'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
        'aws', 'gcp', 'azure', 'vercel', 'netlify', 'heroku',
        'supabase', 'firebase', 'cloudflare',

        # Sync/Storage
        'syncthing', 's3', 'dropbox', 'gdrive',
    }

    # Known organizations
    KNOWN_ORGS: Set[str] = {
        'anthropic', 'openai', 'google', 'microsoft', 'meta', 'facebook',
        'amazon', 'aws', 'apple', 'nvidia', 'amd', 'intel',
        'huggingface', 'hugging face', 'stability', 'stability ai',
        'mistral', 'mistral ai', 'cohere', 'deepmind',
        'github', 'gitlab', 'atlassian', 'jetbrains',
        'mozilla', 'linux foundation', 'apache', 'cncf',
    }

    # Technical concepts to extract
    KNOWN_CONCEPTS: Set[str] = {
        # AI/ML concepts
        'embedding', 'embeddings', 'vector', 'vectors', 'similarity',
        'semantic', 'token', 'tokens', 'tokenization', 'attention',
        'transformer', 'llm', 'nlp', 'rag', 'fine-tuning', 'inference',
        'prompt', 'prompting', 'context window', 'agent', 'agents',

        # Memory/Data concepts
        'memory', 'recall', 'search', 'index', 'query', 'filter',
        'cache', 'persistence', 'storage', 'database', 'schema',
        'migration', 'backup', 'sync', 'replication',

        # Dev concepts
        'api', 'sdk', 'cli', 'gui', 'tui', 'repl',
        'async', 'concurrent', 'parallel', 'thread', 'process',
        'hook', 'plugin', 'extension', 'middleware',
        'config', 'configuration', 'settings', 'environment',

        # Architecture
        'microservice', 'monolith', 'serverless', 'edge',
        'frontend', 'backend', 'fullstack', 'devops', 'mlops',
    }

    # Common words to exclude from project detection
    EXCLUDE_FROM_PROJECTS: Set[str] = {
        'built-in', 'opt-in', 'opt-out', 'log-in', 'sign-in', 'sign-up',
        'set-up', 'break-down', 'follow-up', 'check-in', 'check-out',
        'e-mail', 'e-commerce', 'well-known', 'real-time', 'open-source',
        'end-to-end', 'out-of-the-box', 'plug-and-play', 'point-to-point',
        'self-hosted', 'self-service', 'user-friendly', 'read-only',
        'write-only', 'read-write', 'key-value', 'name-value',
        'non-null', 'non-empty', 'non-zero', 'pre-built', 'pre-configured',
        'post-install', 'pre-install', 'co-authored',
    }

    def __init__(self, custom_tools: Optional[Set[str]] = None,
                 custom_orgs: Optional[Set[str]] = None):
        """Initialize extractor with optional custom entity sets."""
        self.tools = self.KNOWN_TOOLS.copy()
        self.orgs = self.KNOWN_ORGS.copy()
        self.concepts = self.KNOWN_CONCEPTS.copy()

        if custom_tools:
            self.tools.update(custom_tools)
        if custom_orgs:
            self.orgs.update(custom_orgs)

    def extract(self, content: str) -> List[Entity]:
        """Extract all entities from content.

        Returns deduplicated list of entities found in the text.
        """
        entities = []

        # Extract in order of specificity
        entities.extend(self._extract_organizations(content))
        entities.extend(self._extract_tools(content))
        entities.extend(self._extract_projects(content))
        entities.extend(self._extract_concepts(content))
        entities.extend(self._extract_people(content))

        # Deduplicate by normalized name + type
        return self._dedupe(entities)

    def _normalize(self, name: str) -> str:
        """Normalize entity name for comparison and storage."""
        # Lowercase and replace spaces with hyphens
        normalized = name.lower().strip()
        normalized = re.sub(r'\s+', '-', normalized)
        # Remove special characters except hyphens
        normalized = re.sub(r'[^a-z0-9\-]', '', normalized)
        return normalized

    def _extract_projects(self, content: str) -> List[Entity]:
        """Extract project names (kebab-case identifiers)."""
        entities = []

        # Pattern for kebab-case: word-word or word-word-word etc.
        # Must start with letter, contain at least one hyphen
        pattern = r'\b([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\b'

        for match in re.finditer(pattern, content.lower()):
            name = match.group(1)

            # Skip excluded common phrases
            if name in self.EXCLUDE_FROM_PROJECTS:
                continue

            # Skip if it's actually a known tool
            if name in self.tools:
                continue

            # Skip very short matches (like a-b)
            if len(name) < 5:
                continue

            entities.append(Entity(
                name=name,
                normalized=self._normalize(name),
                type='project'
            ))

        return entities

    def _extract_tools(self, content: str) -> List[Entity]:
        """Extract known tools from content."""
        entities = []
        content_lower = content.lower()

        # Find word boundaries for each known tool
        for tool in self.tools:
            # Handle tools with special characters like 'c++', 'c#'
            # For these, use lookahead/lookbehind instead of \b
            if any(c in tool for c in ['+', '#', '.']):
                # Use negative lookbehind/ahead for word chars
                escaped_tool = re.escape(tool)
                pattern = rf'(?<![a-zA-Z0-9]){escaped_tool}(?![a-zA-Z0-9])'
            else:
                escaped_tool = re.escape(tool)
                pattern = rf'\b{escaped_tool}\b'

            if re.search(pattern, content_lower):
                entities.append(Entity(
                    name=tool,
                    normalized=self._normalize(tool),
                    type='tool'
                ))

        return entities

    def _extract_organizations(self, content: str) -> List[Entity]:
        """Extract known organizations from content."""
        entities = []
        content_lower = content.lower()

        for org in self.orgs:
            escaped_org = re.escape(org)
            pattern = rf'\b{escaped_org}\b'

            if re.search(pattern, content_lower):
                entities.append(Entity(
                    name=org.title(),  # Capitalize properly
                    normalized=self._normalize(org),
                    type='organization'
                ))

        return entities

    def _extract_concepts(self, content: str) -> List[Entity]:
        """Extract technical concepts from content."""
        entities = []
        content_lower = content.lower()

        for concept in self.concepts:
            escaped = re.escape(concept)
            pattern = rf'\b{escaped}\b'

            if re.search(pattern, content_lower):
                entities.append(Entity(
                    name=concept,
                    normalized=self._normalize(concept),
                    type='concept'
                ))

        return entities

    def _extract_people(self, content: str) -> List[Entity]:
        """Extract person names from content.

        Looks for:
        - Capitalized words that could be names
        - Special handling for 'user', 'User'
        """
        entities = []

        # Handle 'user' as a person entity
        if re.search(r'\buser\b', content, re.IGNORECASE):
            entities.append(Entity(
                name='User',
                normalized='user',
                type='person'
            ))

        # Look for capitalized name patterns
        # Pattern: One or two capitalized words that look like names
        # Avoid matching at sentence starts by requiring preceding text or punctuation

        # Common first names to boost confidence
        common_names = {
            'john', 'james', 'robert', 'michael', 'william', 'david', 'richard',
            'joseph', 'thomas', 'charles', 'mary', 'patricia', 'jennifer',
            'linda', 'elizabeth', 'barbara', 'susan', 'jessica', 'sarah',
            'alex', 'chris', 'sam', 'taylor', 'jordan', 'casey', 'morgan',
        }

        # Find capitalized words that might be names
        # Look for: word followed by another capitalized word (full name)
        # Or standalone capitalized word that matches common names
        name_pattern = r'(?<=[^\.\!\?\n])\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'

        for match in re.finditer(name_pattern, content):
            potential_name = match.group(1)

            # Skip if it's a known tool or org
            if self._normalize(potential_name) in self.tools:
                continue
            if self._normalize(potential_name) in self.orgs:
                continue

            # Skip common words that might be capitalized
            skip_words = {
                'The', 'This', 'That', 'These', 'Those', 'What', 'When',
                'Where', 'Which', 'Who', 'Why', 'How', 'If', 'Then',
                'But', 'And', 'Or', 'Not', 'For', 'With', 'From',
                'After', 'Before', 'During', 'About', 'Into', 'Through',
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                'Saturday', 'Sunday', 'January', 'February', 'March',
                'April', 'May', 'June', 'July', 'August', 'September',
                'October', 'November', 'December',
            }

            if potential_name in skip_words:
                continue

            # Check if first word is a common name (higher confidence)
            first_word = potential_name.split()[0].lower()
            if first_word in common_names or ' ' in potential_name:
                entities.append(Entity(
                    name=potential_name,
                    normalized=self._normalize(potential_name),
                    type='person'
                ))

        return entities

    def _dedupe(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities by normalized name + type."""
        seen = set()
        unique = []

        for entity in entities:
            key = (entity.normalized, entity.type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique


class EntityStore:
    """Store and manage entities in the database."""

    def __init__(self, db_conn):
        """Initialize with database connection."""
        self.conn = db_conn

    def store_entity(self, entity: Entity) -> int:
        """Store or update an entity, return entity ID."""
        now = int(datetime.now().timestamp())

        # Try to insert, update mention count if exists
        cursor = self.conn.execute("""
            INSERT INTO entities (name, normalized, type, first_seen, last_seen, mention_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(normalized, type) DO UPDATE SET
                last_seen = excluded.last_seen,
                mention_count = mention_count + 1
            RETURNING id
        """, (entity.name, entity.normalized, entity.type, now, now))

        row = cursor.fetchone()
        if row:
            return row[0]

        # Fallback: get existing ID
        cursor = self.conn.execute("""
            SELECT id FROM entities WHERE normalized = ? AND type = ?
        """, (entity.normalized, entity.type))
        row = cursor.fetchone()
        return row[0] if row else -1

    def link_memory_entities(self, memory_id: int, entity_ids: List[int]) -> int:
        """Link a memory to entities. Returns count of links created."""
        count = 0
        for entity_id in entity_ids:
            try:
                self.conn.execute("""
                    INSERT OR IGNORE INTO memory_entities (memory_id, entity_id)
                    VALUES (?, ?)
                """, (memory_id, entity_id))
                count += 1
            except Exception:
                pass
        return count

    def get_memory_entities(self, memory_id: int) -> List[dict]:
        """Get all entities linked to a memory."""
        rows = self.conn.execute("""
            SELECT e.* FROM entities e
            JOIN memory_entities me ON e.id = me.entity_id
            WHERE me.memory_id = ?
        """, (memory_id,)).fetchall()

        return [{
            'id': row['id'],
            'name': row['name'],
            'normalized': row['normalized'],
            'type': row['type'],
            'first_seen': row['first_seen'],
            'last_seen': row['last_seen'],
            'mention_count': row['mention_count'],
        } for row in rows]

    def get_entity(self, entity_id: int) -> Optional[dict]:
        """Get a specific entity by ID."""
        row = self.conn.execute("""
            SELECT * FROM entities WHERE id = ?
        """, (entity_id,)).fetchone()

        if not row:
            return None

        return {
            'id': row['id'],
            'name': row['name'],
            'normalized': row['normalized'],
            'type': row['type'],
            'first_seen': row['first_seen'],
            'last_seen': row['last_seen'],
            'mention_count': row['mention_count'],
        }

    def search_entities(
        self,
        query: Optional[str] = None,
        type: Optional[EntityType] = None,
        limit: int = 50
    ) -> List[dict]:
        """Search entities by name or type."""
        sql = "SELECT * FROM entities WHERE 1=1"
        params = []

        if query:
            sql += " AND (name LIKE ? OR normalized LIKE ?)"
            like_query = f"%{query}%"
            params.extend([like_query, like_query])

        if type:
            sql += " AND type = ?"
            params.append(type)

        sql += " ORDER BY mention_count DESC, last_seen DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [{
            'id': row['id'],
            'name': row['name'],
            'normalized': row['normalized'],
            'type': row['type'],
            'first_seen': row['first_seen'],
            'last_seen': row['last_seen'],
            'mention_count': row['mention_count'],
        } for row in rows]

    def get_memories_for_entity(self, entity_id: int, limit: int = 50) -> List[int]:
        """Get memory IDs linked to an entity."""
        rows = self.conn.execute("""
            SELECT memory_id FROM memory_entities
            WHERE entity_id = ?
            LIMIT ?
        """, (entity_id, limit)).fetchall()

        return [row['memory_id'] for row in rows]

    def list_by_type(self, type: EntityType, limit: int = 100) -> List[dict]:
        """List all entities of a given type."""
        rows = self.conn.execute("""
            SELECT * FROM entities
            WHERE type = ?
            ORDER BY mention_count DESC, last_seen DESC
            LIMIT ?
        """, (type, limit)).fetchall()

        return [{
            'id': row['id'],
            'name': row['name'],
            'normalized': row['normalized'],
            'type': row['type'],
            'first_seen': row['first_seen'],
            'last_seen': row['last_seen'],
            'mention_count': row['mention_count'],
        } for row in rows]

    def get_entity_stats(self) -> dict:
        """Get statistics about stored entities."""
        rows = self.conn.execute("""
            SELECT type, COUNT(*) as count
            FROM entities
            GROUP BY type
        """).fetchall()

        by_type = {row['type']: row['count'] for row in rows}

        total = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

        return {
            'total': total,
            'by_type': by_type,
        }


# Module-level convenience
_extractor: Optional[EntityExtractor] = None


def get_extractor() -> EntityExtractor:
    """Get or create the entity extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def extract_entities(content: str) -> List[Entity]:
    """Extract entities from content using default extractor."""
    return get_extractor().extract(content)
