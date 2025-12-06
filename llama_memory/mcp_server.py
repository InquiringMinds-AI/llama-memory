"""
MCP server for llama-memory.
Exposes memory operations as tools for Claude Code and other MCP clients.
"""

import json
import sys
from typing import Any, Optional

from .config import get_config
from .memory import MemoryStore, get_store

JSONRPC_VERSION = "2.0"
SERVER_NAME = "llama-memory"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"


def send_response(id: Any, result: Any = None, error: Any = None):
    """Send a JSON-RPC response."""
    response = {"jsonrpc": JSONRPC_VERSION, "id": id}
    if error:
        response["error"] = error
    else:
        response["result"] = result
    print(json.dumps(response), flush=True)


def send_notification(method: str, params: Any = None):
    """Send a JSON-RPC notification."""
    notification = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params:
        notification["params"] = params
    print(json.dumps(notification), flush=True)


TOOLS = [
    {
        "name": "memory_store",
        "description": "Store a new memory with semantic embedding. Use this to remember facts, decisions, events, or any information that should persist across sessions. Automatically detects duplicates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store. Be specific and include context."
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "event", "entity", "context", "procedure"],
                    "description": "Type of memory. fact=persistent truth, decision=why something was done, event=what happened, entity=project/person/system, context=working state, procedure=how to do something",
                    "default": "fact"
                },
                "project": {
                    "type": "string",
                    "description": "Project name if memory is project-specific. Omit for global memories."
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Importance from 1-10. Default 5. Use 8+ for critical decisions or key facts.",
                    "default": 5
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization"
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary for quick scanning"
                },
                "retention": {
                    "type": "string",
                    "enum": ["permanent", "long-term", "medium", "short", "session"],
                    "description": "How long to keep. permanent=forever, long-term=5yr, medium=1yr, short=30d",
                    "default": "long-term"
                },
                "force": {
                    "type": "boolean",
                    "description": "Store even if duplicate detected",
                    "default": False
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "memory_search",
        "description": "Search memories by semantic similarity with hybrid ranking. Combines vector distance with importance, recency, and access patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for. Can be a question or keywords."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return.",
                    "default": 10
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "event", "entity", "context", "procedure"],
                    "description": "Filter by memory type"
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project (also includes global memories)"
                },
                "session_id": {
                    "type": "string",
                    "description": "Filter by session ID to see memories from a specific session"
                },
                "min_importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Only return memories with at least this importance"
                },
                "hybrid_ranking": {
                    "type": "boolean",
                    "description": "Use hybrid ranking (distance + importance + recency + access). Default true.",
                    "default": True
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_search_text",
        "description": "Fast full-text search. Less semantic but faster. Good for exact phrase matching.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for"
                },
                "limit": {
                    "type": "integer",
                    "default": 10
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "event", "entity", "context", "procedure"]
                },
                "project": {
                    "type": "string"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_list",
        "description": "List recent memories without search. Good for seeing what's been stored.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 20
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "event", "entity", "context", "procedure"]
                },
                "project": {
                    "type": "string"
                },
                "order_by": {
                    "type": "string",
                    "enum": ["created_at", "updated_at", "accessed_at", "importance"],
                    "default": "created_at"
                }
            }
        }
    },
    {
        "name": "memory_get",
        "description": "Get a specific memory by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Memory ID"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "memory_update",
        "description": "Update an existing memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Memory ID to update"
                },
                "content": {
                    "type": "string",
                    "description": "New content (will re-embed)"
                },
                "summary": {
                    "type": "string"
                },
                "importance": {
                    "type": "integer"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "archived": {
                    "type": "boolean",
                    "description": "Set to true to archive (hide from search)"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "memory_supersede",
        "description": "Create a new memory that replaces an old one. Use when information has changed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "old_id": {
                    "type": "integer",
                    "description": "ID of memory being replaced"
                },
                "new_content": {
                    "type": "string",
                    "description": "The updated information"
                },
                "type": {"type": "string"},
                "project": {"type": "string"},
                "importance": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["old_id", "new_content"]
        }
    },
    {
        "name": "memory_delete",
        "description": "Delete (archive) a memory. Use hard=true to permanently delete.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer"
                },
                "hard": {
                    "type": "boolean",
                    "description": "Permanently delete instead of archiving",
                    "default": False
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "memory_stats",
        "description": "Get statistics about stored memories.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "memory_export",
        "description": "Export all memories as JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_archived": {
                    "type": "boolean",
                    "default": False
                }
            }
        }
    },
    {
        "name": "memory_cleanup",
        "description": "Archive memories past their retention period.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "memory_recall",
        "description": "Get a context block of relevant memories for session start. Returns high-importance memories formatted for context injection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional topic to focus recall on"
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project"
                },
                "limit": {
                    "type": "integer",
                    "default": 10
                },
                "min_importance": {
                    "type": "integer",
                    "description": "Minimum importance (default 7 for auto, 5 with query)"
                },
                "format": {
                    "type": "string",
                    "enum": ["compact", "markdown", "json"],
                    "default": "compact"
                }
            }
        }
    },
    {
        "name": "memory_related",
        "description": "Find memories related to a specific memory (similar content, same session, supersession chain).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Memory ID to find relations for"
                },
                "limit": {
                    "type": "integer",
                    "default": 10
                }
            },
            "required": ["id"]
        }
    }
]


def handle_initialize(id: Any, params: dict):
    """Handle initialize request."""
    send_response(id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        }
    })


def handle_list_tools(id: Any):
    """Handle tools/list request."""
    send_response(id, {"tools": TOOLS})


def handle_call_tool(id: Any, params: dict, store: MemoryStore):
    """Handle tools/call request."""
    tool_name = params.get("name")
    args = params.get("arguments", {})

    try:
        result_text = ""

        if tool_name == "memory_store":
            memory_id, duplicate = store.store(
                content=args["content"],
                type=args.get("type", "fact"),
                summary=args.get("summary"),
                project=args.get("project"),
                importance=args.get("importance", 5),
                tags=args.get("tags", []),
                retention=args.get("retention", "long-term"),
                force=args.get("force", False),
            )
            if memory_id == -1 and duplicate:
                result_text = f"Duplicate detected! Similar memory exists (ID: {duplicate.existing_id}, similarity: {duplicate.similarity:.1%}). Use force=true to store anyway."
            elif duplicate:
                result_text = f"Stored memory with ID {memory_id} (note: similar to #{duplicate.existing_id})"
            else:
                result_text = f"Stored memory with ID {memory_id}"

        elif tool_name == "memory_search":
            results = store.search(
                query=args["query"],
                limit=args.get("limit", 10),
                type=args.get("type"),
                project=args.get("project"),
                session_id=args.get("session_id"),
                min_importance=args.get("min_importance"),
                hybrid_ranking=args.get("hybrid_ranking", True),
            )
            result_text = json.dumps([m.to_dict() for m in results], indent=2)

        elif tool_name == "memory_search_text":
            results = store.search_text(
                query=args["query"],
                limit=args.get("limit", 10),
                type=args.get("type"),
                project=args.get("project"),
            )
            result_text = json.dumps([m.to_dict() for m in results], indent=2)

        elif tool_name == "memory_list":
            results = store.list(
                limit=args.get("limit", 20),
                type=args.get("type"),
                project=args.get("project"),
                order_by=args.get("order_by", "created_at"),
            )
            result_text = json.dumps([m.to_dict() for m in results], indent=2)

        elif tool_name == "memory_get":
            memory = store.get(args["id"])
            if memory:
                result_text = json.dumps(memory.to_dict(), indent=2)
            else:
                result_text = f"Memory {args['id']} not found"

        elif tool_name == "memory_update":
            store.update(
                id=args["id"],
                content=args.get("content"),
                summary=args.get("summary"),
                importance=args.get("importance"),
                tags=args.get("tags"),
                archived=args.get("archived"),
            )
            result_text = f"Updated memory {args['id']}"

        elif tool_name == "memory_supersede":
            new_id = store.supersede(
                old_id=args["old_id"],
                new_content=args["new_content"],
                type=args.get("type"),
                project=args.get("project"),
                importance=args.get("importance"),
                tags=args.get("tags"),
            )
            result_text = f"Created memory {new_id} superseding {args['old_id']}"

        elif tool_name == "memory_delete":
            store.delete(args["id"], hard=args.get("hard", False))
            action = "Deleted" if args.get("hard") else "Archived"
            result_text = f"{action} memory {args['id']}"

        elif tool_name == "memory_stats":
            result_text = json.dumps(store.stats(), indent=2)

        elif tool_name == "memory_export":
            data = store.export_json(include_archived=args.get("include_archived", False))
            result_text = json.dumps(data, indent=2)

        elif tool_name == "memory_cleanup":
            count = store.cleanup_expired()
            result_text = f"Archived {count} expired memories"

        elif tool_name == "memory_recall":
            query = args.get("query")
            limit = args.get("limit", 10)
            project = args.get("project")
            min_importance = args.get("min_importance")
            fmt = args.get("format", "compact")

            if query:
                memories = store.search(
                    query=query,
                    limit=limit,
                    project=project,
                    min_importance=min_importance or 5,
                )
            else:
                memories = store.list(
                    limit=limit,
                    project=project,
                    order_by='accessed_at',
                )
                memories = [m for m in memories if m.importance >= (min_importance or 7)]

            if fmt == "markdown":
                lines = ["# Memory Context\n"]
                for m in memories:
                    lines.append(f"## [{m.type.upper()}] (importance: {m.importance})")
                    if m.project:
                        lines.append(f"*Project: {m.project}*\n")
                    lines.append(m.content)
                    lines.append("")
                result_text = "\n".join(lines)
            elif fmt == "compact":
                lines = []
                for m in memories:
                    prefix = f"[{m.type}:{m.importance}]"
                    if m.project:
                        prefix += f" ({m.project})"
                    content = m.content[:150] + "..." if len(m.content) > 150 else m.content
                    lines.append(f"{prefix} {content}")
                result_text = "\n".join(lines)
            else:
                result_text = json.dumps([m.to_dict() for m in memories], indent=2)

        elif tool_name == "memory_related":
            source = store.get(args["id"])
            if not source:
                result_text = f"Memory {args['id']} not found"
            else:
                limit = args.get("limit", 10)
                results = []

                # Similar memories
                similar = store.search(query=source.content, limit=limit + 1)
                for m in similar:
                    if m.id != source.id:
                        results.append({"memory": m.to_dict(), "relation": "similar"})

                # Same session
                if source.session_id:
                    session_mems = store.search(query=source.content, limit=50, session_id=source.session_id)
                    for m in session_mems:
                        if m.id != source.id and m.id not in [r["memory"]["id"] for r in results]:
                            results.append({"memory": m.to_dict(), "relation": "same-session"})

                # Supersession chain
                if source.superseded_by:
                    newer = store.get(source.superseded_by)
                    if newer:
                        results.append({"memory": newer.to_dict(), "relation": "supersedes-this"})

                conn = store.db.conn
                older_row = conn.execute("SELECT id FROM memories WHERE superseded_by = ?", (source.id,)).fetchone()
                if older_row:
                    older = store.get(older_row['id'])
                    if older:
                        results.append({"memory": older.to_dict(), "relation": "superseded-by-this"})

                results = results[:limit]
                result_text = json.dumps(results, indent=2)

        else:
            send_response(id, error={"code": -32601, "message": f"Unknown tool: {tool_name}"})
            return

        send_response(id, {
            "content": [{"type": "text", "text": result_text}]
        })

    except Exception as e:
        send_response(id, error={"code": -32000, "message": str(e)})


def run_server():
    """Run the MCP server."""
    config = get_config()

    # Validate configuration
    errors = config.validate()
    if errors:
        sys.stderr.write(f"Configuration errors:\n")
        for error in errors:
            sys.stderr.write(f"  - {error}\n")
        sys.stderr.write("Run 'llama-memory install' to set up dependencies.\n")
        sys.exit(1)

    # Initialize store
    store = get_store(config)
    store.initialize()

    # Process JSON-RPC messages
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get("method")
            id = request.get("id")
            params = request.get("params", {})

            if method == "initialize":
                handle_initialize(id, params)
            elif method == "notifications/initialized":
                pass  # Client acknowledgment
            elif method == "tools/list":
                handle_list_tools(id)
            elif method == "tools/call":
                handle_call_tool(id, params, store)
            else:
                if id is not None:
                    send_response(id, error={"code": -32601, "message": f"Unknown method: {method}"})

        except json.JSONDecodeError as e:
            send_response(None, error={"code": -32700, "message": f"Parse error: {e}"})
        except Exception as e:
            send_response(None, error={"code": -32000, "message": str(e)})


if __name__ == "__main__":
    run_server()
