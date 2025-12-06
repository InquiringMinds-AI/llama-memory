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

- **Semantic search** - Find related memories even without exact keyword matches
- **Hybrid ranking** - Combines vector distance with importance, recency, and access patterns
- **Duplicate detection** - Warns before storing similar content
- **Session tracking** - Groups memories by conversation session
- **Memory decay** - Auto-archives old conversational memories while protecting project knowledge
- **Hierarchical memories** - Parent/child relationships for structured knowledge
- **Conflict detection** - Identifies potentially contradicting memories
- **Fully local** - No internet required after setup
- **Lightweight** - ~21MB embedding model, <150KB sqlite-vec extension
- **MCP server** - Works with Claude Code and any MCP client (34 tools)
- **CLI tool** - Full command-line interface (46 commands)
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
| Linux | x86_64 | ✓ |
| Linux | aarch64 | ✓ |
| macOS | arm64 (M1/M2) | ✓ |
| macOS | x86_64 | ✓ |
| Android (Termux) | aarch64 | ✓ |

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
from llama_memory import store, search, stats

# Store
memory_id = store(
    "Project X uses MIT license for brand building",
    type="decision",
    project="project-x",
    importance=8
)

# Search
results = search("licensing decisions", limit=5)
for memory in results:
    print(f"[{memory.id}] {memory.content} (distance: {memory.distance:.3f})")

# Stats
print(stats())
```

### MCP Server (Claude Code)

Add to your Claude Code settings (`~/.claude/settings.local.json`):

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

Then Claude Code will have access to 34 memory tools:

| Tool | Description |
|------|-------------|
| `memory_store` | Store new memories (with duplicate detection) |
| `memory_search` | Semantic search with hybrid ranking |
| `memory_search_text` | Fast full-text search |
| `memory_list` | List recent memories |
| `memory_get` | Get by ID |
| `memory_update` | Update existing |
| `memory_supersede` | Replace outdated info |
| `memory_delete` | Archive or delete (with cascade option) |
| `memory_unarchive` | Restore archived memory |
| `memory_stats` | Statistics |
| `memory_export` | Export to JSON |
| `memory_export_md` | Export to Markdown |
| `memory_export_csv` | Export to CSV |
| `memory_cleanup` | Archive expired memories |
| `memory_recall` | Get context block for session start |
| `memory_related` | Find related memories |
| `memory_projects` | List all projects |
| `memory_sessions` | List/browse sessions |
| `memory_link` | Create link between memories |
| `memory_unlink` | Remove link |
| `memory_links` | Get links for a memory |
| `memory_types` | List memory types |
| `memory_count` | Count memories |
| `memory_topics` | List/get/set topics |
| `memory_merge` | Merge multiple memories |
| `memory_children` | Get child memories |
| `memory_parent` | Get/set parent memory |
| `memory_confidence` | Set confidence level |
| `memory_conflicts` | Check for conflicts |
| `memory_decay` | Decay status and run |
| `memory_protect` | Protect from decay |
| `memory_tags` | List all tags |
| `memory_history` | Memory access history |
| `memory_search_history` | Search query patterns |

## Memory Types

| Type | Use For |
|------|---------|
| `fact` | Persistent truths ("User prefers privacy-first tools") |
| `decision` | Why something was done ("Chose MIT for brand building") |
| `event` | What happened when ("Released v1.0 on 2025-01-15") |
| `entity` | Projects, people, systems |
| `context` | Current working state |
| `procedure` | How to do something |

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

### Database Schema (v6)

- `memories` - Main table with content, type, project, importance, retention, confidence, parent_id, topic, protected
- `memory_embeddings` - Vector embeddings (sqlite-vec virtual table)
- `memories_fts` - Full-text search index (SQLite FTS5)
- `memory_links` - Explicit relationships between memories (with weights)
- `memory_log` - Audit log for debugging
- `embedding_versions` - Track embeddings across model migrations
- `search_history` - Query pattern tracking
- `topics` - Topic definitions
- `memory_conflicts` - Potential contradiction tracking
- `meta` - System metadata (decay runs, etc.)

## Performance

On a Snapdragon 8 Elite (Samsung Galaxy S25):
- Embedding generation: ~30-50ms per memory
- Semantic search: <100ms for 10 results
- Database size: ~1KB per memory (with embedding)

## Comparison

| Feature | llama-memory | mcp-memory-service | mcp-memory-keeper |
|---------|--------------|-------------------|-------------------|
| Local embeddings | ✓ llama.cpp | ONNX (x86 only) | None |
| Runs on Android | ✓ | ✗ | ✓ |
| Semantic search | ✓ | ✓ | ✗ |
| Dependencies | Minimal | Heavy (PyTorch optional) | Minimal |
| MCP server | ✓ | ✓ | ✓ |

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
