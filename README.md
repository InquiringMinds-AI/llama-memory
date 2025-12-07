# llama-memory

**Local vector memory for AI assistants.** No cloud APIs. No heavy dependencies. Runs on phones.

Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for embeddings and [sqlite-vec](https://github.com/asg017/sqlite-vec) for similarity search. Everything runs locally.

## Why?

Existing memory solutions for AI assistants either:
- Require cloud APIs for embeddings (OpenAI, etc.)
- Need heavy ML dependencies (PyTorch, ONNX) that don't work everywhere
- Use simple keyword matching that misses semantic connections

llama-memory solves this by using llama.cpp, which compiles to a single binary and runs on almost anything - including Android phones via Termux.

## Features

### Core Features
- **Semantic search** - Find related memories even without exact keyword matches
- **Hybrid ranking** - Combines vector distance with importance, recency, and access patterns
- **Duplicate detection** - Warns before storing similar content
- **Conflict detection** - Identifies potentially contradicting memories
- **Session tracking** - Groups memories by conversation session
- **Memory decay** - Auto-archives old conversational memories while protecting project knowledge
- **Hierarchical memories** - Parent/child relationships for structured knowledge

### v2.7 Features (NEW)
- **Session persistence** - Save and resume work sessions across context resets
- **Task progress tracking** - Track completed and pending tasks within sessions
- **Session decisions/next steps** - Capture key decisions and what to do next
- **Resume prompts** - Generate context blocks for seamless session resumption
- **Auto-capture hooks** - Claude Code hooks for automatic session save/restore
- **Transcript parsing** - Extract session state from Claude Code conversation history

### v2.6 Features
- **Full config.yaml system** - All settings configurable via YAML
- **Configurable scoring weights** - Tune semantic, importance, recency, frequency, confidence, entity_match weights
- **Entity match scoring bonus** - Memories matching query entities rank higher
- **Enhanced name extraction** - Detects "Dr. Johnson", "by John Smith", possessive forms like "Alex's"
- **PDF ingestion** - Extract and chunk PDF documents (optional: `pip install llama-memory[pdf]`)

### v2.5 Features
- **Token-budgeted recall** - Dual-response pattern: all summaries + top memories within token budget
- **Auto-summarization** - Automatically generates summaries on store
- **Entity extraction** - Automatically extracts people, projects, tools, concepts, organizations
- **Named contexts** - Organize memories by context (work, personal, project-specific)
- **Document ingestion** - Ingest Markdown, text, JSON files with smart chunking
- **JSONL export/import** - Git-friendly export format with merge strategies
- **Auto-capture modes** - Green (all), brown (sessions), quiet (minimal) capture modes

### Platform & Integration
- **Fully local** - No internet required after setup
- **Lightweight** - ~21MB embedding model, <150KB sqlite-vec extension
- **MCP server** - Works with Claude Code and any MCP client (44 tools)
- **CLI tool** - Full command-line interface (60+ commands)
- **Python API** - Import and use directly
- **Runs on Android** - Works in Termux with Snapdragon 8 Elite and similar hardware
- **5-year retention** - Built for long-term memory with configurable retention policies

## Requirements

- Python 3.10+
- CMake and make (for building llama.cpp)
- ~500MB disk space for llama.cpp build
- ~25MB for model and sqlite-vec

### Tested Platforms

| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux | x86_64 | Works |
| Linux | aarch64 | Works |
| macOS | arm64 (M1/M2) | Works |
| macOS | x86_64 | Works |
| Android (Termux) | aarch64 | Works |

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/InquiringMinds-AI/llama-memory.git
cd llama-memory

# Run the install script (builds llama.cpp, downloads model, installs sqlite-vec)
./scripts/install.sh

# Install the Python package
pip install -e .

# Initialize the database
llama-memory init

# Verify everything works
llama-memory doctor
```

### Manual Install

If you prefer to set things up manually:

1. **Build llama.cpp:**
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   mkdir build && cd build
   cmake ..
   make llama-embedding
   ```

2. **Download embedding model:**
   ```bash
   curl -L -o all-MiniLM-L6-v2-Q4_K_M.gguf \
     "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf"
   ```

3. **Download sqlite-vec:**
   ```bash
   # For Linux x86_64:
   curl -L -o sqlite-vec.tar.gz \
     "https://github.com/asg017/sqlite-vec/releases/download/v0.1.6/sqlite-vec-0.1.6-loadable-linux-x86_64.tar.gz"
   tar -xzf sqlite-vec.tar.gz
   ```

4. **Install llama-memory:**
   ```bash
   pip install -e .
   ```

5. **Configure paths** (optional - auto-discovery usually works):
   ```bash
   mkdir -p ~/.config/llama-memory
   cat > ~/.config/llama-memory/config.yaml << EOF
   database_path: ~/.local/share/llama-memory/memory.db
   embedding_binary: /path/to/llama.cpp/build/bin/llama-embedding
   embedding_model: /path/to/all-MiniLM-L6-v2-Q4_K_M.gguf
   sqlite_vec_path: /path/to/vec0.so
   EOF
   ```

## Usage

### CLI

```bash
# Store a memory
llama-memory store "User prefers dark mode and F-Droid apps" --type fact --importance 8

# Store with duplicate detection (warns if similar exists)
llama-memory store "User likes dark mode" --type fact
# Use --force to store anyway

# Semantic search
llama-memory search "what apps does the user like"

# Search with date filters
llama-memory search "deployments" --after 2025-01-01 --before 2025-06-01
llama-memory search "recent bugs" --after 7d   # last 7 days
llama-memory search "this month" --after 1m    # relative dates: 7d, 1w, 1m

# List recent memories
llama-memory list --limit 10
llama-memory list --after 1w --project myapi

# Browse projects and sessions
llama-memory projects
llama-memory sessions
llama-memory sessions session-1234567890  # view specific session

# Get statistics
llama-memory stats

# Find related memories
llama-memory related 42  # finds similar, same-session, supersession chain

# Export all memories
llama-memory export -o backup.json
```

### Memory Management

```bash
# Update a memory
llama-memory update 42 --importance 9

# Archive (soft delete) and restore
llama-memory delete 42
llama-memory unarchive 42

# Delete with children (cascade)
llama-memory delete 42 --cascade           # archive parent and all children
llama-memory delete 42 --hard --cascade    # permanently delete parent and children

# Replace outdated information
llama-memory supersede 42 "Updated information here"

# Backup and restore
llama-memory backup
llama-memory backup --output /path/to/backup.db
llama-memory restore /path/to/backup.db

# Re-embed all memories (after model upgrade)
llama-memory reembed --yes
llama-memory reembed --old-model "previous-model-name" --yes
```

### Python API

```python
from llama_memory import store, search, stats, get_store

# Basic usage
memory_id = store(
    "Project X uses MIT license for brand building",
    type="decision",
    project="project-x",
    importance=8
)

results = search("licensing decisions", limit=5)
for memory in results:
    print(f"[{memory.id}] {memory.content} (distance: {memory.distance:.3f})")

# v2.5: Token-budgeted recall
store_instance = get_store()
response = store_instance.recall_budgeted(query="project decisions", max_tokens=2000)
print(f"Total matches: {response.total_matches}")
print(f"Full memories: {len(response.full_memories)}")
print(f"Tokens used: {response.tokens_used}/{response.tokens_budget}")

# v2.5: Entity extraction
from llama_memory import extract_entities
entities = extract_entities("Using Python with SQLite and MCP protocol")
for e in entities:
    print(f"[{e.type}] {e.name}")

# v2.5: Document ingestion
from llama_memory import ingest_document
memory_ids = ingest_document("document.md", importance=6)

# v2.5: Named contexts
store_instance.create_context("work", "Work-related memories")
store_instance.store(content="Meeting notes...", context="work")
```

### MCP Server (Claude Code)

Add to your Claude Code settings (`~/.claude.json` under `projects` or global `mcpServers`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "llama-memory",
      "args": ["serve"],
      "env": {
        "LD_LIBRARY_PATH": "/path/to/llama.cpp/build/bin"
      }
    }
  }
}
```

Then Claude Code will have access to 44 memory tools:

#### Core Operations
| Tool | Description |
|------|-------------|
| `memory_store` | Store new memories (with duplicate detection, auto-summarization, entity extraction) |
| `memory_search` | Semantic search with hybrid ranking |
| `memory_search_text` | Fast full-text search |
| `memory_list` | List recent memories |
| `memory_get` | Get by ID |
| `memory_update` | Update existing |
| `memory_supersede` | Replace outdated info |
| `memory_delete` | Archive or delete (with cascade option) |
| `memory_unarchive` | Restore archived memory |
| `memory_stats` | Statistics |

#### Export & Recall
| Tool | Description |
|------|-------------|
| `memory_recall` | Get context block (supports token budgeting) |
| `memory_export` | Export to JSON |
| `memory_export_md` | Export to Markdown |
| `memory_export_csv` | Export to CSV |
| `memory_export_jsonl` | Export to JSONL (git-friendly) |
| `memory_import_jsonl` | Import from JSONL |

#### Organization
| Tool | Description |
|------|-------------|
| `memory_related` | Find related memories |
| `memory_projects` | List all projects |
| `memory_sessions` | List/browse sessions |
| `memory_contexts` | Manage named contexts (v2.5) |
| `memory_entities` | Search extracted entities (v2.5) |
| `memory_link` | Create link between memories |
| `memory_unlink` | Remove link |
| `memory_links` | Get links for a memory |
| `memory_types` | List memory types |
| `memory_count` | Count memories |
| `memory_topics` | List/get/set topics |
| `memory_tags` | List all tags |

#### Advanced
| Tool | Description |
|------|-------------|
| `memory_merge` | Merge multiple memories |
| `memory_children` | Get child memories |
| `memory_parent` | Get/set parent memory |
| `memory_confidence` | Set confidence level |
| `memory_conflicts` | Check for conflicts |
| `memory_cleanup` | Archive expired memories |
| `memory_decay` | Decay status and run |
| `memory_protect` | Protect from decay |
| `memory_history` | Memory access history |
| `memory_search_history` | Search query patterns |
| `memory_ingest` | Ingest documents (v2.5) |
| `memory_capture` | Auto-capture modes (v2.5) |

#### Session Persistence (v2.7)
| Tool | Description |
|------|-------------|
| `session_save` | Save session state with progress, decisions, next steps |
| `session_resume` | Get resume prompt for continuing a session |
| `session_list` | List recent sessions |
| `session_get` | Get full session details |

## Memory Types

| Type | Use For |
|------|---------|
| `fact` | Persistent truths ("User prefers privacy-first tools") |
| `decision` | Why something was done ("Chose MIT for brand building") |
| `event` | What happened when ("Released v1.0 on 2025-01-15") |
| `entity` | Projects, people, systems |
| `context` | Current working state |
| `procedure` | How to do something |
| `session` | Work session state (v2.7) |

## Retention Policies

| Policy | Duration | Use For |
|--------|----------|---------|
| `permanent` | Forever | Core facts, critical decisions |
| `long-term` | 5 years | Most memories (default) |
| `medium` | 1 year | Project-specific context |
| `short` | 30 days | Temporary notes |
| `session` | Until cleanup | Working context |

Run `llama-memory cleanup` periodically to archive expired memories.

## Memory Decay

The decay system automatically archives old, unused memories while protecting important knowledge:

```bash
# Check decay status
llama-memory decay --status

# Preview what would be archived
llama-memory decay --dry-run

# Run decay (archive old memories)
llama-memory decay --run

# Manually protect a memory
llama-memory protect 42
llama-memory protect 42 --remove  # remove protection
```

**Protected from decay:**
- Memories with a project set
- Memories with parent/child relationships
- Types: entity, decision, procedure
- Importance >= 7
- Retention = permanent
- Memories with links
- Manually protected

**Decay timeline:**
- 120 days without access: starts affecting search ranking
- 180 days without access: auto-archived (if not protected)

## v2.5 Features

### Token-Budgeted Recall

Get all matching memories as summaries, plus full content for top matches within a token budget:

```python
response = store.recall_budgeted(query="topic", max_tokens=4000)
# response.summaries = ALL matching memories as summaries
# response.full_memories = Top memories with full content (within budget)
# response.tokens_used = Actual tokens used
```

### Entity Extraction

Automatically extracts and indexes entities from memory content:

- **Projects**: kebab-case identifiers (llama-memory, claude-code)
- **Tools**: Known dev tools, languages, frameworks (Python, SQLite, MCP)
- **People**: Names and user references
- **Concepts**: Technical terms (API, embedding, vector)
- **Organizations**: Companies (Anthropic, Github)

```bash
# Search entities via CLI
llama-memory entities list
llama-memory entities search "llama"
llama-memory entities type tool
```

### Named Contexts

Organize memories by context:

```python
store.create_context("work", "Work-related memories")
store.store(content="...", context="work")
stats = store.get_context_stats("work")
```

### Document Ingestion

Ingest documents with smart chunking:

```bash
llama-memory ingest document.md --importance 6
llama-memory ingest --list  # List ingested files
```

### JSONL Export/Import

Git-friendly export format:

```bash
llama-memory export-jsonl memories.jsonl
llama-memory import-jsonl memories.jsonl --merge skip
```

### Auto-Capture Modes

Development modes for automatic memory capture:

| Mode | Captures |
|------|----------|
| `green` | Files, commands, sessions, errors |
| `brown` | Sessions and errors only |
| `quiet` | Minimal (default) |

## v2.6 Features

### Full Configuration System

All settings are now configurable via `~/.config/llama-memory/config.yaml`:

```yaml
# Scoring weights (auto-normalized to sum to 1.0)
scoring:
  semantic: 0.40      # Vector similarity
  importance: 0.25    # User-assigned importance
  recency: 0.15       # Time decay
  frequency: 0.10     # Access count
  confidence: 0.05    # Memory certainty
  entity_match: 0.05  # Entity overlap bonus

# Entity extraction settings
entities:
  enabled: true
  extract_names: true  # Enable name heuristics
  max_entities: 20     # Cap per memory
  additional_tools:    # Custom tool names
    - myframework
  additional_orgs:     # Custom org names
    - mycompany

# Document ingestion
ingestion:
  pdf_enabled: true
  max_chunk_size: 500
  overlap: 50

# Token budget defaults
budget:
  max_tokens: 4000
  summary_tokens: 25

# Memory decay
decay:
  enabled: true
  start_days: 120
  archive_days: 180
```

```python
# Access config in Python
from llama_memory import get_config, reload_config

config = get_config()
print(config.scoring.semantic)  # 0.40

# Reload after editing config.yaml
reload_config()
```

### Enhanced Entity Extraction

Name detection now uses heuristics to find people:

```python
from llama_memory import extract_entities

# Detects names after prefixes (Dr., by, from, etc.)
extract_entities("Dr. Johnson said...")  # -> Johnson (person)

# Detects two-word names
extract_entities("by John Smith")  # -> John Smith (person)

# Detects possessive forms
extract_entities("Alex's project")  # -> Alex (person)

# Detects known first names
extract_entities("Sarah mentioned...")  # -> Sarah (person)
```

### PDF Ingestion

Ingest PDF documents (requires optional dependency):

```bash
# Install PDF support
pip install llama-memory[pdf]

# Ingest a PDF
llama-memory ingest research-paper.pdf --importance 7

# Or via Python
from llama_memory import ingest_pdf, is_pdf_available

if is_pdf_available():
    memory_ids = ingest_pdf("paper.pdf", project="research")
```

PDF metadata (title, author) is extracted and stored with chunks.

### Entity Match Scoring

Search results now rank higher if they share entities with your query:

```python
# If query mentions "Claude" and "Anthropic", memories containing
# those entities get a scoring bonus
results = search("Claude and Anthropic announcements")
# Memories about Claude/Anthropic rank higher even with similar vector distances
```

This is controlled by `scoring.entity_match` weight in config.

## v2.7 Features

### Session Persistence

Save and resume work sessions across Claude Code context resets or between conversations:

```bash
# Save current session state
llama-memory session-save "Implementing auth feature" \
  --summary "Adding OAuth2 to the API" \
  --task "Implement OAuth2 authentication" \
  --decision "Using Auth0 for identity provider" \
  --decision "JWT tokens for session management" \
  --next "Add token refresh endpoint" \
  --next "Write integration tests" \
  --blocker "Waiting for Auth0 credentials" \
  --file "src/auth/oauth.py" \
  --file "src/auth/tokens.py" \
  --project "myapi"

# List recent sessions
llama-memory session-list
llama-memory session-list --project myapi

# Get session details
llama-memory session-get 42

# Get resume prompt for latest session
llama-memory session-resume
llama-memory session-resume --project myapi
llama-memory session-resume --id 42
```

The resume prompt generates a structured context block that can be injected into a new conversation:

```
<session-resume>
Resuming session: Implementing auth feature

**Summary:** Adding OAuth2 to the API

**Task:** Implement OAuth2 authentication

**Key decisions made:**
  - Using Auth0 for identity provider
  - JWT tokens for session management

**Next steps:**
  - Add token refresh endpoint
  - Write integration tests

**Blockers:**
  - Waiting for Auth0 credentials
</session-resume>
```

### Session Python API

```python
from llama_memory import (
    save_session,
    resume_session,
    list_sessions,
    get_session,
    Session,
    SessionStore,
)

# Save a session
session = save_session(
    title="Feature implementation",
    summary="Working on user auth",
    task_description="Add OAuth2 support",
    progress={"completed": 3, "total": 7, "items": [
        {"task": "Design auth flow", "done": True},
        {"task": "Create OAuth client", "done": True},
        {"task": "Implement callback", "done": True},
        {"task": "Add token refresh", "done": False},
        {"task": "Write tests", "done": False},
        {"task": "Update docs", "done": False},
        {"task": "Deploy", "done": False},
    ]},
    files_touched=["src/auth.py", "tests/test_auth.py"],
    decisions=["Using Auth0", "JWT for sessions"],
    next_steps=["Add refresh endpoint", "Write tests"],
    blockers=["Need Auth0 credentials"],
    project="myapi",
)
print(f"Session saved: ID {session.id}")

# Get resume prompt
prompt = resume_session(project="myapi")
print(prompt)

# List sessions
for s in list_sessions(limit=5):
    print(f"[{s.id}] {s.title}")

# Get specific session
session = get_session(42)
print(session.to_resume_prompt())
```

### Session MCP Tools

When using the MCP server, four session tools are available:

| Tool | Description |
|------|-------------|
| `session_save` | Save session with title, summary, progress, files, decisions, next steps, blockers |
| `session_resume` | Get resume prompt for session (by ID or latest for project) |
| `session_list` | List recent sessions, optionally filtered by project |
| `session_get` | Get full session details by ID |

### Auto-Capture Hooks (Claude Code Integration)

For automatic session persistence without manual commands, install the Claude Code hooks:

```bash
# Install hooks
llama-memory hooks-install

# Check status
llama-memory hooks-status

# Remove hooks
llama-memory hooks-uninstall
```

This installs three hooks into `~/.claude/settings.json`:

| Hook | Trigger | Action |
|------|---------|--------|
| `PreCompact` | Before context compaction | Auto-saves session state |
| `SessionEnd` | When you exit Claude Code | Auto-saves session state |
| `SessionStart` | When session starts/resumes | Injects session context |

**How it works:**

1. **Before compaction**: When context fills up, Claude Code compacts old messages. The `PreCompact` hook saves your task state (todos, files, decisions) before compaction loses detail.

2. **On exit**: The `SessionEnd` hook saves session state when you close Claude Code.

3. **On start**: The `SessionStart` hook checks for recent sessions and automatically injects resume context, so Claude knows what you were working on.

**Transcript parsing**: The hooks parse Claude Code's conversation transcript (JSONL) to extract:
- Current todo list state (from TodoWrite tool calls)
- Files read/written/edited
- User messages (to infer task description)
- Session ID and metadata

This makes session persistence fully automatic - no manual `session-save` needed.

## Model Changes

If you switch embedding models, use the reembed command:

```bash
# Preview what will be re-embedded
llama-memory reembed

# Re-embed all memories
llama-memory reembed --yes

# Only re-embed memories from old model
llama-memory reembed --old-model "previous-model-name" --yes
```

This regenerates embeddings while preserving the original text.

## Architecture

```
~/.local/share/llama-memory/
├── memory.db          # SQLite database with sqlite-vec extension
├── bin/
│   └── llama-embedding
├── lib/
│   ├── vec0.so
│   └── libllama.so
└── models/
    └── all-MiniLM-L6-v2-Q4_K_M.gguf

~/.config/llama-memory/
└── config.yaml        # Optional configuration overrides
```

### Database Schema (v8)

**Core tables:**
- `memories` - Main table with content, type (now includes 'session'), project, importance, retention, confidence, parent_id, topic, protected, token_count, context, entities, file_path, chunk_of, chunk_index
- `memory_embeddings` - Vector embeddings (sqlite-vec virtual table)
- `memories_fts` - Full-text search index (SQLite FTS5)
- `memory_links` - Explicit relationships between memories (with weights)
- `memory_log` - Audit log for debugging
- `embedding_versions` - Track embeddings across model migrations
- `search_history` - Query pattern tracking
- `memory_conflicts` - Potential contradiction tracking
- `meta` - System metadata (decay runs, capture config, etc.)

**v2.5 tables:**
- `entities` - Extracted entities (name, type, mention_count)
- `memory_entities` - Memory-entity junction table
- `contexts` - Named contexts for organization
- `capture_log` - Auto-capture event log

## Performance

On a Snapdragon 8 Elite (Samsung Galaxy S25):
- Embedding generation: ~30-50ms per memory
- Semantic search: <100ms for 10 results
- Entity extraction: <5ms per memory
- Database size: ~1KB per memory (with embedding)

## Comparison

| Feature | llama-memory | mcp-memory-service | claude-memory-mcp | Memento | mcp-knowledge-graph |
|---------|--------------|-------------------|-------------------|---------|---------------------|
| Local embeddings | llama.cpp | ONNX (x86 only) | Cloud API | Local | Cloud API |
| Runs on Android | Yes | No | No | No | No |
| Semantic search | Yes | Yes | Yes | Yes | Yes |
| Token budgeting | Yes | No | Yes | No | No |
| Entity extraction | Yes | No | No | No | Yes |
| Conflict detection | Yes | No | No | No | No |
| Duplicate detection | Yes | Yes | No | No | No |
| Document ingestion | Yes | No | No | Yes | No |
| JSONL export | Yes | No | No | No | Yes |
| Session persistence | Yes | No | No | No | No |
| MCP tools | 44 | 8 | 12 | 5 | 10 |
| CLI commands | 60+ | 0 | 0 | 0 | 0 |

## Troubleshooting

### "Embedding binary not found"

Run the install script or set `embedding_binary` in config:
```bash
./scripts/install.sh
# or
llama-memory config  # shows current config
```

### "sqlite-vec extension not found"

Download the correct version for your platform:
```bash
# Check your platform
uname -sm
# Download from https://github.com/asg017/sqlite-vec/releases
```

### Slow embedding generation

The first embedding takes longer (model loading). Subsequent ones are faster.
If consistently slow, check:
- Model is on fast storage (not SD card)
- No thermal throttling

### Run health check

```bash
llama-memory doctor
```

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## Credits

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Inference engine
- [sqlite-vec](https://github.com/asg017/sqlite-vec) - Vector search extension
- [All-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Embedding model

Built by [InquiringMinds-AI](https://github.com/InquiringMinds-AI).
