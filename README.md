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
- **Fully local** - No internet required after setup
- **Lightweight** - ~21MB embedding model, <150KB sqlite-vec extension
- **MCP server** - Works with Claude Code and any MCP client
- **CLI tool** - Full command-line interface
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

# Semantic search
llama-memory search "what apps does the user like"

# List recent memories
llama-memory list --limit 10

# Get statistics
llama-memory stats

# Export all memories
llama-memory export -o backup.json
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

Then Claude Code will have access to these tools:
- `memory_store` - Store new memories
- `memory_search` - Semantic similarity search
- `memory_search_text` - Fast full-text search
- `memory_list` - List recent memories
- `memory_get` - Get by ID
- `memory_update` - Update existing
- `memory_supersede` - Replace outdated info
- `memory_delete` - Archive or delete
- `memory_stats` - Statistics
- `memory_export` - Export to JSON
- `memory_cleanup` - Archive expired memories

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

## Model Changes

If you switch embedding models, run the migration script:

```bash
python scripts/migrate.py
```

This re-embeds all memories with the new model while preserving the original text.

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

### Database Schema

- `memories` - Main table with content, type, project, importance, retention
- `memory_embeddings` - Vector embeddings (sqlite-vec virtual table)
- `memories_fts` - Full-text search index (SQLite FTS5)
- `memory_log` - Audit log for debugging
- `embedding_models` - Track model versions for migrations

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
