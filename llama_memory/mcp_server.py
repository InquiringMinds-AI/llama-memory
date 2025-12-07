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
                },
                "context": {
                    "type": "string",
                    "description": "Named context for this memory (work, personal, etc.). Used to organize memories."
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
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Token budget for response. If set, uses dual-response pattern: returns ALL summaries + full content for top memories within budget.",
                    "default": None
                },
                "context": {
                    "type": "string",
                    "description": "Filter by named context (work, personal, etc.)"
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
    },
    # ========== v2.5 Tools ==========
    {
        "name": "memory_contexts",
        "description": "Manage named contexts for organizing memories. Contexts allow grouping memories by topic, project, or any category.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "get", "delete", "set_default"],
                    "description": "Action to perform: list all contexts, create new, get stats, delete, or set default",
                    "default": "list"
                },
                "name": {
                    "type": "string",
                    "description": "Context name (required for create, get, delete, set_default)"
                },
                "description": {
                    "type": "string",
                    "description": "Context description (optional, for create)"
                },
                "migrate_to": {
                    "type": "string",
                    "description": "When deleting, migrate memories to this context instead of clearing"
                }
            }
        }
    },
    {
        "name": "memory_entities",
        "description": "Search and explore extracted entities (people, projects, tools, concepts, organizations). Entities are automatically extracted from memory content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "search", "stats", "memories"],
                    "description": "Action: list entities, search by name, get stats, or get memories for an entity",
                    "default": "list"
                },
                "query": {
                    "type": "string",
                    "description": "Search query for entity name (for search action)"
                },
                "type": {
                    "type": "string",
                    "enum": ["person", "project", "tool", "concept", "organization", "location"],
                    "description": "Filter by entity type"
                },
                "entity_id": {
                    "type": "integer",
                    "description": "Entity ID (for memories action)"
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum results to return"
                }
            }
        }
    },
    {
        "name": "memory_ingest",
        "description": "Ingest a document file into memory. Supports Markdown, text, and JSON. Large files are automatically chunked into linked memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ingest", "list", "delete"],
                    "description": "Action: ingest a file, list ingested files, or delete ingested file",
                    "default": "ingest"
                },
                "path": {
                    "type": "string",
                    "description": "File path to ingest or filter by"
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "event", "entity", "context", "procedure"],
                    "description": "Memory type for chunks",
                    "default": "fact"
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Importance level for chunks",
                    "default": 5
                },
                "project": {
                    "type": "string",
                    "description": "Project to associate with"
                },
                "context": {
                    "type": "string",
                    "description": "Named context for memories"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to apply to all chunks"
                },
                "max_chunk_size": {
                    "type": "integer",
                    "description": "Maximum characters per chunk",
                    "default": 500
                },
                "hard_delete": {
                    "type": "boolean",
                    "description": "Permanently delete instead of archive (for delete action)",
                    "default": False
                }
            }
        }
    },
    {
        "name": "memory_export_jsonl",
        "description": "Export memories to JSONL file (git-friendly, one JSON per line).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Output file path"
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived memories",
                    "default": False
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "memory_import_jsonl",
        "description": "Import memories from JSONL file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Input file path"
                },
                "merge_strategy": {
                    "type": "string",
                    "enum": ["skip", "overwrite", "newer"],
                    "description": "How to handle duplicates: skip, overwrite, or keep newer",
                    "default": "skip"
                },
                "context": {
                    "type": "string",
                    "description": "Context to assign to imported memories"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "memory_capture",
        "description": "Manage auto-capture modes for automatic memory creation. Modes: green (capture everything), brown (sessions only), quiet (minimal).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "set_mode", "process", "log_discovery", "recent", "config"],
                    "description": "Action: get status, set mode, process pending events, log a discovery, view recent events, or update config",
                    "default": "status"
                },
                "mode": {
                    "type": "string",
                    "enum": ["green", "brown", "quiet"],
                    "description": "Capture mode (for set_mode action)"
                },
                "content": {
                    "type": "string",
                    "description": "Discovery content to log (for log_discovery action)"
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Importance for discovery (for log_discovery action)",
                    "default": 5
                },
                "limit": {
                    "type": "integer",
                    "description": "Limit for process or recent actions",
                    "default": 50
                },
                "event_type": {
                    "type": "string",
                    "enum": ["file_change", "command", "session_start", "session_end", "error", "discovery"],
                    "description": "Filter by event type (for recent action)"
                },
                "capture_files": {
                    "type": "boolean",
                    "description": "Enable file capture (for config action)"
                },
                "capture_commands": {
                    "type": "boolean",
                    "description": "Enable command capture (for config action)"
                },
                "capture_sessions": {
                    "type": "boolean",
                    "description": "Enable session capture (for config action)"
                },
                "auto_context": {
                    "type": "string",
                    "description": "Default context for auto-captured memories (for config action)"
                }
            }
        }
    },
    # Session persistence tools
    {
        "name": "session_save",
        "description": "Save the current session state for later resumption. Use before context overflow or when switching tasks. Captures task progress, files, decisions, and next steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the session (e.g., 'Refactoring auth system')"
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of what was being worked on"
                },
                "task_description": {
                    "type": "string",
                    "description": "Detailed description of the task"
                },
                "progress": {
                    "type": "object",
                    "description": "Progress tracking: {completed: N, total: M, items: [{task: 'x', done: true}, ...]}"
                },
                "files_touched": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of files that were read or modified"
                },
                "decisions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key decisions made during the session"
                },
                "next_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What to do next when resuming"
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What's blocking progress"
                },
                "project": {
                    "type": "string",
                    "description": "Project name"
                },
                "notes": {
                    "type": "string",
                    "description": "Additional freeform notes"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization"
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "session_resume",
        "description": "Get a context block to resume a previous session. Returns formatted prompt with task state, progress, decisions, and next steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "Specific session ID to resume. If omitted, gets the most recent session."
                },
                "project": {
                    "type": "string",
                    "description": "Filter by project when getting most recent session"
                }
            }
        }
    },
    {
        "name": "session_list",
        "description": "List recent saved sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Filter by project"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of sessions to return",
                    "default": 10
                }
            }
        }
    },
    {
        "name": "session_get",
        "description": "Get details of a specific session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "Session ID to retrieve"
                }
            },
            "required": ["session_id"]
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
                context=args.get("context"),
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
            max_tokens = args.get("max_tokens")
            context = args.get("context")

            # Use token-budgeted recall if max_tokens specified
            if max_tokens:
                response = store.recall_budgeted(
                    query=query,
                    limit=limit,
                    max_tokens=max_tokens,
                    project=project,
                    context=context,
                    min_importance=min_importance,
                )
                result_text = json.dumps(response.to_dict(), indent=2)
            else:
                # Legacy recall behavior
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

                # Filter by context if specified
                if context:
                    memories = [m for m in memories if getattr(m, 'context', None) == context or getattr(m, 'context', None) is None]

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

        # ========== v2.5 Tool Handlers ==========

        elif tool_name == "memory_contexts":
            action = args.get("action", "list")

            if action == "list":
                contexts = store.list_contexts()
                default = store.get_default_context()
                result_text = json.dumps({
                    "contexts": contexts,
                    "default": default
                }, indent=2)

            elif action == "create":
                name = args.get("name")
                if not name:
                    result_text = "Error: name required for create action"
                else:
                    ctx_id = store.create_context(name, args.get("description"))
                    if ctx_id == -1:
                        result_text = f"Error: Context '{name}' already exists"
                    else:
                        result_text = f"Created context '{name}' with ID {ctx_id}"

            elif action == "get":
                name = args.get("name")
                if not name:
                    result_text = "Error: name required for get action"
                else:
                    stats = store.get_context_stats(name)
                    result_text = json.dumps(stats, indent=2)

            elif action == "delete":
                name = args.get("name")
                if not name:
                    result_text = "Error: name required for delete action"
                else:
                    count = store.delete_context(name, migrate_to=args.get("migrate_to"))
                    migrate_note = f" (migrated to '{args.get('migrate_to')}')" if args.get("migrate_to") else ""
                    result_text = f"Deleted context '{name}', affected {count} memories{migrate_note}"

            elif action == "set_default":
                name = args.get("name")
                if not name:
                    result_text = "Error: name required for set_default action"
                else:
                    if store.set_default_context(name):
                        result_text = f"Set '{name}' as default context"
                    else:
                        result_text = f"Error: Context '{name}' not found"

            else:
                result_text = f"Error: Unknown action '{action}'"

        elif tool_name == "memory_entities":
            action = args.get("action", "list")
            limit = args.get("limit", 50)

            if action == "list":
                entities = store.get_entities(type=args.get("type"), limit=limit)
                result_text = json.dumps(entities, indent=2)

            elif action == "search":
                query = args.get("query")
                if not query:
                    result_text = "Error: query required for search action"
                else:
                    entities = store.get_entities(query=query, type=args.get("type"), limit=limit)
                    result_text = json.dumps(entities, indent=2)

            elif action == "stats":
                stats = store.get_entity_stats()
                result_text = json.dumps(stats, indent=2)

            elif action == "memories":
                entity_id = args.get("entity_id")
                if not entity_id:
                    result_text = "Error: entity_id required for memories action"
                else:
                    memories = store.get_entity_memories(entity_id, limit=limit)
                    result_text = json.dumps([m.to_dict() for m in memories], indent=2)

            else:
                result_text = f"Error: Unknown action '{action}'"

        elif tool_name == "memory_ingest":
            from .ingester import Ingester
            ingester = Ingester(store)
            action = args.get("action", "ingest")

            if action == "ingest":
                path = args.get("path")
                if not path:
                    result_text = "Error: path required for ingest action"
                else:
                    try:
                        memory_ids = ingester.ingest(
                            path=path,
                            type=args.get("type", "fact"),
                            importance=args.get("importance", 5),
                            project=args.get("project"),
                            context=args.get("context"),
                            tags=args.get("tags"),
                            max_chunk_size=args.get("max_chunk_size"),
                        )
                        result_text = f"Ingested {path}: created {len(memory_ids)} memory chunks (IDs: {memory_ids[:5]}{'...' if len(memory_ids) > 5 else ''})"
                    except FileNotFoundError:
                        result_text = f"Error: File not found: {path}"
                    except Exception as e:
                        result_text = f"Error ingesting file: {e}"

            elif action == "list":
                files = ingester.list_ingested(path=args.get("path"))
                result_text = json.dumps(files, indent=2)

            elif action == "delete":
                path = args.get("path")
                if not path:
                    result_text = "Error: path required for delete action"
                else:
                    count = ingester.delete_ingested(path, hard=args.get("hard_delete", False))
                    result_text = f"Deleted {count} memory chunks from {path}"

            else:
                result_text = f"Error: Unknown action '{action}'"

        elif tool_name == "memory_export_jsonl":
            path = args.get("path")
            if not path:
                result_text = "Error: path required"
            else:
                try:
                    count = store.export_jsonl(
                        path=path,
                        include_archived=args.get("include_archived", False)
                    )
                    result_text = f"Exported {count} memories to {path}"
                except Exception as e:
                    result_text = f"Error exporting: {e}"

        elif tool_name == "memory_import_jsonl":
            path = args.get("path")
            if not path:
                result_text = "Error: path required"
            else:
                try:
                    stats = store.import_jsonl(
                        path=path,
                        merge_strategy=args.get("merge_strategy", "skip"),
                        context=args.get("context")
                    )
                    result_text = f"Import complete: {stats['imported']} imported, {stats['skipped']} skipped, {stats['updated']} updated, {stats['errors']} errors"
                except FileNotFoundError:
                    result_text = f"Error: File not found: {path}"
                except Exception as e:
                    result_text = f"Error importing: {e}"

        elif tool_name == "memory_capture":
            from .capture import CaptureEngine
            engine = CaptureEngine(store)
            action = args.get("action", "status")

            if action == "status":
                status = engine.get_status()
                result_text = json.dumps(status, indent=2)

            elif action == "set_mode":
                mode = args.get("mode")
                if not mode:
                    result_text = "Error: mode required for set_mode action"
                else:
                    config = engine.set_mode(mode)
                    result_text = f"Capture mode set to '{mode}'. Files: {config.capture_files}, Commands: {config.capture_commands}, Sessions: {config.capture_sessions}"

            elif action == "process":
                limit = args.get("limit", 50)
                stats = engine.process_pending(limit=limit)
                result_text = f"Processed {stats['processed']} events: {stats['stored']} stored, {stats['skipped']} skipped, {stats['errors']} errors"

            elif action == "log_discovery":
                content = args.get("content")
                if not content:
                    result_text = "Error: content required for log_discovery action"
                else:
                    importance = args.get("importance", 5)
                    log_id = engine.log_discovery(content, importance=importance)
                    result_text = f"Logged discovery (ID: {log_id}). Use process action to store as memory."

            elif action == "recent":
                limit = args.get("limit", 50)
                event_type = args.get("event_type")
                events = engine.get_recent_events(limit=limit, event_type=event_type)
                result_text = json.dumps(events, indent=2)

            elif action == "config":
                # Update specific config values
                updates = {}
                for key in ['capture_files', 'capture_commands', 'capture_sessions', 'auto_context']:
                    if args.get(key) is not None:
                        updates[key] = args[key]

                if updates:
                    config = engine.update_config(**updates)
                    result_text = f"Config updated to custom mode: {json.dumps(config.to_dict(), indent=2)}"
                else:
                    result_text = json.dumps(engine.config.to_dict(), indent=2)

            else:
                result_text = f"Error: Unknown action '{action}'"

        # Session persistence tools
        elif tool_name == "session_save":
            from .session import SessionStore
            session_store = SessionStore(store)
            session = session_store.save(
                title=args["title"],
                summary=args.get("summary"),
                task_description=args.get("task_description"),
                progress=args.get("progress"),
                files_touched=args.get("files_touched"),
                decisions=args.get("decisions"),
                next_steps=args.get("next_steps"),
                blockers=args.get("blockers"),
                project=args.get("project"),
                notes=args.get("notes"),
                tags=args.get("tags"),
            )
            result_text = f"Session saved with ID {session.id}: {session.title}"

        elif tool_name == "session_resume":
            from .session import SessionStore
            session_store = SessionStore(store)
            resume_prompt = session_store.resume(
                session_id=args.get("session_id"),
                project=args.get("project"),
            )
            if resume_prompt:
                result_text = resume_prompt
            else:
                result_text = "No session found to resume."

        elif tool_name == "session_list":
            from .session import SessionStore
            session_store = SessionStore(store)
            sessions = session_store.list(
                project=args.get("project"),
                limit=args.get("limit", 10),
            )
            result_list = []
            for s in sessions:
                result_list.append({
                    "id": s.id,
                    "title": s.title,
                    "summary": s.summary,
                    "project": s.project,
                    "ended_at": s.ended_at,
                    "next_steps": s.next_steps[:3] if s.next_steps else [],
                })
            result_text = json.dumps(result_list, indent=2)

        elif tool_name == "session_get":
            from .session import SessionStore
            session_store = SessionStore(store)
            session = session_store.get(args["session_id"])
            if session:
                result_text = json.dumps({
                    "id": session.id,
                    "title": session.title,
                    "summary": session.summary,
                    "task_description": session.task_description,
                    "progress": session.progress,
                    "files_touched": session.files_touched,
                    "decisions": session.decisions,
                    "next_steps": session.next_steps,
                    "blockers": session.blockers,
                    "project": session.project,
                    "notes": session.notes,
                    "started_at": session.started_at,
                    "ended_at": session.ended_at,
                }, indent=2)
            else:
                result_text = f"Session {args['session_id']} not found"

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
    # Use readline() instead of for-in iteration to avoid buffered read-ahead issues
    while True:
        line = sys.stdin.readline()
        if not line:
            break
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
