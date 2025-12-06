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
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata to store with the memory (JSON object)"
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
        "description": "Delete (archive) a memory. Use hard=true to permanently delete. Use cascade=true to also delete child memories.",
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
                },
                "cascade": {
                    "type": "boolean",
                    "description": "Also delete/archive child memories (and their children recursively)",
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
    },
    {
        "name": "memory_projects",
        "description": "List all projects with memory counts and last activity date.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "memory_sessions",
        "description": "List recent sessions or get memories from a specific session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "If provided, return memories from this specific session"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of sessions or memories to return",
                    "default": 20
                }
            }
        }
    },
    {
        "name": "memory_unarchive",
        "description": "Restore an archived memory back to active status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Memory ID to unarchive"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "memory_link",
        "description": "Create an explicit link between two memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "integer",
                    "description": "Source memory ID"
                },
                "target_id": {
                    "type": "integer",
                    "description": "Target memory ID"
                },
                "link_type": {
                    "type": "string",
                    "description": "Type of relationship (related, depends_on, contradicts, etc.)",
                    "default": "related"
                },
                "note": {
                    "type": "string",
                    "description": "Optional note about the relationship"
                },
                "weight": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Relationship strength (0.0-1.0, default 1.0)",
                    "default": 1.0
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata for the link (JSON object)"
                }
            },
            "required": ["source_id", "target_id"]
        }
    },
    {
        "name": "memory_unlink",
        "description": "Remove a link between two memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "integer",
                    "description": "Source memory ID"
                },
                "target_id": {
                    "type": "integer",
                    "description": "Target memory ID"
                }
            },
            "required": ["source_id", "target_id"]
        }
    },
    {
        "name": "memory_links",
        "description": "Get all links for a memory (both incoming and outgoing).",
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
        "name": "memory_types",
        "description": "List all memory types in use with counts.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "memory_count",
        "description": "Count memories with optional filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Filter by memory type"
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project"
                },
                "min_importance": {
                    "type": "integer",
                    "description": "Minimum importance level"
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived memories",
                    "default": False
                }
            }
        }
    },
    # ========== v2.0 Tools ==========
    {
        "name": "memory_topics",
        "description": "List all topics or get/set topic for a memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "get", "set"],
                    "description": "Action: list all topics, get memories by topic, or set topic for memory",
                    "default": "list"
                },
                "topic": {
                    "type": "string",
                    "description": "Topic name (for get/set)"
                },
                "memory_id": {
                    "type": "integer",
                    "description": "Memory ID (for set)"
                },
                "limit": {
                    "type": "integer",
                    "default": 50
                }
            }
        }
    },
    {
        "name": "memory_merge",
        "description": "Merge multiple memories into one new memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Memory IDs to merge (at least 2)"
                },
                "content": {
                    "type": "string",
                    "description": "Content for the merged memory"
                },
                "archive_sources": {
                    "type": "boolean",
                    "description": "Archive the source memories after merging",
                    "default": False
                }
            },
            "required": ["ids", "content"]
        }
    },
    {
        "name": "memory_children",
        "description": "Get child memories of a parent memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Parent memory ID"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "memory_parent",
        "description": "Get ancestors or set parent of a memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Memory ID"
                },
                "set_parent": {
                    "type": "integer",
                    "description": "Parent ID to set (0 or omit to just get ancestors)"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "memory_confidence",
        "description": "Set confidence level for a memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Memory ID"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence level (0.0-1.0)"
                }
            },
            "required": ["id", "confidence"]
        }
    },
    {
        "name": "memory_conflicts",
        "description": "Check for or list memory conflicts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "check_id": {
                    "type": "integer",
                    "description": "Memory ID to check for conflicts"
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold for conflict detection",
                    "default": 0.7
                },
                "conflict_type": {
                    "type": "string",
                    "enum": ["potential", "confirmed", "resolved"],
                    "description": "Filter by conflict type when listing"
                }
            }
        }
    },
    {
        "name": "memory_export_md",
        "description": "Export all memories as Markdown.",
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
        "name": "memory_export_csv",
        "description": "Export all memories as CSV.",
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
    # ========== v2.1 Tools ==========
    {
        "name": "memory_decay",
        "description": "Check decay status or run decay process to archive old, unused memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "check", "run"],
                    "description": "status=system overview, check=check specific memory, run=execute decay",
                    "default": "status"
                },
                "memory_id": {
                    "type": "integer",
                    "description": "Memory ID (for check action)"
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview without archiving (for run action)",
                    "default": True
                }
            }
        }
    },
    {
        "name": "memory_protect",
        "description": "Manually protect or unprotect a memory from decay.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Memory ID"
                },
                "protected": {
                    "type": "boolean",
                    "description": "True to protect, False to remove protection",
                    "default": True
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "memory_tags",
        "description": "List all tags in use with memory counts.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "memory_history",
        "description": "Get access/modification history for a specific memory.",
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
        "name": "memory_search_history",
        "description": "Get search history and popular queries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["recent", "popular"],
                    "description": "recent=recent searches, popular=most frequent queries",
                    "default": "recent"
                },
                "limit": {
                    "type": "integer",
                    "default": 20
                },
                "session_id": {
                    "type": "string",
                    "description": "Filter by session (for recent action)"
                }
            }
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
                source='mcp',
                metadata=args.get("metadata"),
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
            store.delete(args["id"], hard=args.get("hard", False), cascade=args.get("cascade", False))
            action = "Deleted" if args.get("hard") else "Archived"
            cascade_note = " (with children)" if args.get("cascade") else ""
            result_text = f"{action} memory {args['id']}{cascade_note}"

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

        elif tool_name == "memory_projects":
            conn = store.db.conn
            rows = conn.execute("""
                SELECT project, COUNT(*) as count,
                       MAX(importance) as max_importance,
                       MAX(created_at) as latest
                FROM memories
                WHERE archived = 0 AND project IS NOT NULL AND project != ''
                GROUP BY project
                ORDER BY count DESC
            """).fetchall()

            projects = {}
            for row in rows:
                projects[row['project']] = {
                    'count': row['count'],
                    'max_importance': row['max_importance'],
                    'latest': row['latest']
                }
            result_text = json.dumps(projects, indent=2)

        elif tool_name == "memory_sessions":
            conn = store.db.conn
            session_id = args.get("session_id")
            limit = args.get("limit", 20)

            if session_id:
                # Get memories from specific session
                rows = conn.execute("""
                    SELECT id, content, type, importance, created_at
                    FROM memories
                    WHERE session_id = ? AND archived = 0
                    ORDER BY created_at ASC
                """, (session_id,)).fetchall()

                memories = [
                    {
                        'id': row['id'],
                        'content': row['content'],
                        'type': row['type'],
                        'importance': row['importance'],
                        'created_at': row['created_at']
                    }
                    for row in rows
                ]
                result_text = json.dumps({"session_id": session_id, "memories": memories}, indent=2)
            else:
                # List sessions
                rows = conn.execute("""
                    SELECT session_id, COUNT(*) as count,
                           MIN(created_at) as started,
                           MAX(created_at) as ended
                    FROM memories
                    WHERE session_id IS NOT NULL AND archived = 0
                    GROUP BY session_id
                    ORDER BY started DESC
                    LIMIT ?
                """, (limit,)).fetchall()

                sessions = [
                    {
                        'session_id': row['session_id'],
                        'count': row['count'],
                        'started': row['started'],
                        'ended': row['ended']
                    }
                    for row in rows
                ]
                result_text = json.dumps(sessions, indent=2)

        elif tool_name == "memory_unarchive":
            success = store.unarchive(args["id"])
            if success:
                result_text = f"Unarchived memory {args['id']}"
            else:
                result_text = f"Memory {args['id']} not found"

        elif tool_name == "memory_link":
            try:
                created = store.link(
                    source_id=args["source_id"],
                    target_id=args["target_id"],
                    link_type=args.get("link_type", "related"),
                    note=args.get("note"),
                    weight=args.get("weight", 1.0),
                    metadata=args.get("metadata"),
                )
                if created:
                    result_text = f"Linked memory {args['source_id']} -> {args['target_id']}"
                else:
                    result_text = f"Link already exists between {args['source_id']} and {args['target_id']}"
            except ValueError as e:
                result_text = f"Error: {e}"

        elif tool_name == "memory_unlink":
            removed = store.unlink(args["source_id"], args["target_id"])
            if removed:
                result_text = f"Removed link between {args['source_id']} and {args['target_id']}"
            else:
                result_text = f"No link found between {args['source_id']} and {args['target_id']}"

        elif tool_name == "memory_links":
            links = store.get_links(args["id"])
            result_text = json.dumps(links, indent=2)

        elif tool_name == "memory_types":
            conn = store.db.conn
            rows = conn.execute("""
                SELECT type, COUNT(*) as count
                FROM memories
                WHERE archived = 0
                GROUP BY type
                ORDER BY count DESC
            """).fetchall()
            types = {row['type']: row['count'] for row in rows}
            result_text = json.dumps(types, indent=2)

        elif tool_name == "memory_count":
            conn = store.db.conn
            sql = "SELECT COUNT(*) as cnt FROM memories WHERE 1=1"
            params = []

            if not args.get("include_archived", False):
                sql += " AND archived = 0"

            if args.get("type"):
                sql += " AND type = ?"
                params.append(args["type"])

            if args.get("project"):
                sql += " AND project = ?"
                params.append(args["project"])

            if args.get("min_importance"):
                sql += " AND importance >= ?"
                params.append(args["min_importance"])

            row = conn.execute(sql, params).fetchone()
            result_text = str(row['cnt'])

        # ========== v2.0 Tool Handlers ==========

        elif tool_name == "memory_topics":
            action = args.get("action", "list")
            if action == "list":
                topics = store.get_topics()
                result_text = json.dumps(topics, indent=2)
            elif action == "get":
                topic = args.get("topic")
                if not topic:
                    result_text = "Error: topic required for get action"
                else:
                    mems = store.get_by_topic(topic, limit=args.get("limit", 50))
                    result_text = json.dumps([m.to_dict() for m in mems], indent=2)
            elif action == "set":
                memory_id = args.get("memory_id")
                topic = args.get("topic")
                if not memory_id or not topic:
                    result_text = "Error: memory_id and topic required for set action"
                else:
                    if store.set_topic(memory_id, topic):
                        result_text = f"Set topic '{topic}' for memory {memory_id}"
                    else:
                        result_text = f"Memory {memory_id} not found"

        elif tool_name == "memory_merge":
            try:
                merged_id = store.merge(
                    source_ids=args["ids"],
                    merged_content=args["content"],
                    archive_sources=args.get("archive_sources", False),
                )
                result_text = f"Created merged memory {merged_id} from {args['ids']}"
            except ValueError as e:
                result_text = f"Error: {e}"

        elif tool_name == "memory_children":
            children = store.get_children(args["id"])
            result_text = json.dumps([c.to_dict() for c in children], indent=2)

        elif tool_name == "memory_parent":
            if "set_parent" in args:
                parent_id = args["set_parent"] if args["set_parent"] > 0 else None
                try:
                    if store.set_parent(args["id"], parent_id):
                        if parent_id:
                            result_text = f"Set parent of memory {args['id']} to {parent_id}"
                        else:
                            result_text = f"Removed parent from memory {args['id']}"
                    else:
                        result_text = f"Memory {args['id']} not found"
                except ValueError as e:
                    result_text = f"Error: {e}"
            else:
                ancestors = store.get_ancestors(args["id"])
                result_text = json.dumps([a.to_dict() for a in ancestors], indent=2)

        elif tool_name == "memory_confidence":
            try:
                if store.set_confidence(args["id"], args["confidence"]):
                    result_text = f"Set confidence of memory {args['id']} to {args['confidence']}"
                else:
                    result_text = f"Memory {args['id']} not found"
            except ValueError as e:
                result_text = f"Error: {e}"

        elif tool_name == "memory_conflicts":
            if args.get("check_id"):
                conflicts = store.check_conflicts(args["check_id"], threshold=args.get("threshold", 0.7))
                result_text = json.dumps(conflicts, indent=2)
            else:
                conflicts = store.get_conflicts(conflict_type=args.get("conflict_type"))
                result_text = json.dumps(conflicts, indent=2)

        elif tool_name == "memory_export_md":
            result_text = store.export_markdown(include_archived=args.get("include_archived", False))

        elif tool_name == "memory_export_csv":
            result_text = store.export_csv(include_archived=args.get("include_archived", False))

        # ========== v2.1 Tool Handlers ==========

        elif tool_name == "memory_decay":
            from .database import DECAY_START_DAYS, DECAY_ARCHIVE_DAYS
            action = args.get("action", "status")

            if action == "status":
                last_run = store.get_last_decay_run()
                candidates = store.get_decay_candidates()
                result_text = json.dumps({
                    "decay_start_days": DECAY_START_DAYS,
                    "decay_archive_days": DECAY_ARCHIVE_DAYS,
                    "last_run": last_run,
                    "candidates_count": len(candidates),
                }, indent=2)
            elif action == "check":
                memory_id = args.get("memory_id")
                if not memory_id:
                    result_text = "Error: memory_id required for check action"
                else:
                    status = store.get_decay_status(memory_id)
                    result_text = json.dumps(status, indent=2)
            elif action == "run":
                dry_run = args.get("dry_run", True)
                result = store.run_decay(dry_run=dry_run)
                result_text = json.dumps(result, indent=2)

        elif tool_name == "memory_protect":
            protected = args.get("protected", True)
            if store.set_protected(args["id"], protected):
                if protected:
                    result_text = f"Memory {args['id']} is now protected from decay"
                else:
                    result_text = f"Memory {args['id']} protection removed"
            else:
                result_text = f"Memory {args['id']} not found"

        elif tool_name == "memory_tags":
            tags = store.get_tags()
            result_text = json.dumps(tags, indent=2)

        elif tool_name == "memory_history":
            history = store.get_access_history(args["id"])
            result_text = json.dumps(history, indent=2)

        elif tool_name == "memory_search_history":
            action = args.get("action", "recent")
            limit = args.get("limit", 20)
            if action == "recent":
                history = store.get_search_history(limit=limit, session_id=args.get("session_id"))
                result_text = json.dumps(history, indent=2)
            else:  # popular
                popular = store.get_popular_queries(limit=limit)
                result_text = json.dumps(popular, indent=2)

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
