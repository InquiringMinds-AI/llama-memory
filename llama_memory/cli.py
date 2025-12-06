"""
Command-line interface for llama-memory.
"""

import argparse
import json
import sys
from pathlib import Path

from .config import get_config, ensure_directories, Config, DATA_DIR, CONFIG_FILE
from .database import get_database
from .memory import get_store, MemoryStore


def cmd_init(args):
    """Initialize the memory database."""
    config = get_config()
    errors = config.validate()

    if errors:
        print("Configuration issues:")
        for error in errors:
            print(f"  - {error}")
        print("\nRun 'llama-memory install' to set up dependencies.")
        return 1

    store = get_store(config)
    store.initialize()
    print(f"Database initialized at {config.database_path}")
    return 0


def cmd_store(args):
    """Store a new memory."""
    config = get_config()
    store = get_store(config)
    store.initialize()

    tags = args.tag if args.tag else []

    memory_id, duplicate = store.store(
        content=args.content,
        type=args.type,
        project=args.project,
        importance=args.importance,
        tags=tags,
        summary=args.summary,
        retention=args.retention,
        force=args.force,
    )

    if memory_id == -1 and duplicate:
        print(f"Duplicate detected! Similar memory exists:")
        print(f"  ID: {duplicate.existing_id}")
        print(f"  Similarity: {duplicate.similarity:.1%}")
        print(f"  Content: {duplicate.existing_content[:150]}...")
        print(f"\nUse --force to store anyway, or 'supersede {duplicate.existing_id}' to update.")
        return 1

    if duplicate:
        print(f"Stored memory {memory_id} (note: similar to #{duplicate.existing_id})")
    else:
        print(f"Stored memory {memory_id}")
    return 0


def cmd_search(args):
    """Search memories."""
    config = get_config()
    store = get_store(config)

    results = store.search(
        query=args.query,
        limit=args.limit,
        type=args.type,
        project=args.project,
        session_id=args.session,
        min_importance=args.min_importance,
        hybrid_ranking=not args.no_hybrid,
    )

    if args.format == 'json':
        print(json.dumps([m.to_dict() for m in results], indent=2))
    else:
        for m in results:
            score_str = f"score={m.score:.4f}" if m.score is not None else f"distance={m.distance:.4f}"
            print(f"\n[{m.id}] ({m.type}) importance={m.importance} {score_str}")
            if m.project:
                print(f"    project: {m.project}")
            if m.session_id:
                print(f"    session: {m.session_id}")
            print(f"    {m.content[:200]}{'...' if len(m.content) > 200 else ''}")
    return 0


def cmd_search_text(args):
    """Full-text search memories."""
    config = get_config()
    store = get_store(config)

    results = store.search_text(
        query=args.query,
        limit=args.limit,
        type=args.type,
        project=args.project,
    )

    if args.format == 'json':
        print(json.dumps([m.to_dict() for m in results], indent=2))
    else:
        for m in results:
            print(f"\n[{m.id}] ({m.type}) importance={m.importance}")
            if m.project:
                print(f"    project: {m.project}")
            print(f"    {m.content[:200]}{'...' if len(m.content) > 200 else ''}")
    return 0


def cmd_list(args):
    """List memories."""
    config = get_config()
    store = get_store(config)

    results = store.list(
        limit=args.limit,
        type=args.type,
        project=args.project,
        order_by=args.order_by,
        include_archived=args.archived,
    )

    if args.format == 'json':
        print(json.dumps([m.to_dict() for m in results], indent=2))
    else:
        for m in results:
            print(f"\n[{m.id}] ({m.type}) importance={m.importance}")
            if m.project:
                print(f"    project: {m.project}")
            print(f"    {m.content[:200]}{'...' if len(m.content) > 200 else ''}")
    return 0


def cmd_get(args):
    """Get a specific memory."""
    config = get_config()
    store = get_store(config)

    memory = store.get(args.id)
    if memory:
        print(json.dumps(memory.to_dict(), indent=2))
    else:
        print(f"Memory {args.id} not found")
        return 1
    return 0


def cmd_update(args):
    """Update a memory."""
    config = get_config()
    store = get_store(config)

    store.update(
        id=args.id,
        content=args.content,
        summary=args.summary,
        importance=args.importance,
        archived=args.archive,
    )
    print(f"Updated memory {args.id}")
    return 0


def cmd_supersede(args):
    """Supersede a memory."""
    config = get_config()
    store = get_store(config)

    new_id = store.supersede(args.old_id, args.content)
    print(f"Created memory {new_id} superseding {args.old_id}")
    return 0


def cmd_delete(args):
    """Delete a memory."""
    config = get_config()
    store = get_store(config)

    store.delete(args.id, hard=args.hard)
    action = "Deleted" if args.hard else "Archived"
    print(f"{action} memory {args.id}")
    return 0


def cmd_stats(args):
    """Show memory statistics."""
    config = get_config()
    store = get_store(config)

    stats = store.stats()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_export(args):
    """Export memories to JSON."""
    config = get_config()
    store = get_store(config)

    data = store.export_json(include_archived=args.archived)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(data)} memories to {args.output}")
    else:
        print(json.dumps(data, indent=2))
    return 0


def cmd_cleanup(args):
    """Clean up expired memories."""
    config = get_config()
    store = get_store(config)

    count = store.cleanup_expired()
    print(f"Archived {count} expired memories")
    return 0


def cmd_config(args):
    """Show or edit configuration."""
    config = get_config()

    if args.show:
        print(json.dumps(config.to_dict(), indent=2, default=str))
    elif args.path:
        print(CONFIG_FILE)
    else:
        # Show config with validation
        print("Current configuration:")
        print(json.dumps(config.to_dict(), indent=2, default=str))
        print()
        errors = config.validate()
        if errors:
            print("Issues:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("Configuration OK")
    return 0


def cmd_doctor(args):
    """Check system health."""
    config = get_config()

    print("=== llama-memory health check ===\n")

    # Configuration
    print("Configuration:")
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"  [X] {error}")
    else:
        print(f"  [OK] Embedding binary: {config.embedding_binary}")
        print(f"  [OK] Embedding model: {config.embedding_model}")
        print(f"  [OK] sqlite-vec: {config.sqlite_vec_path}")

    # Database
    print("\nDatabase:")
    if config.database_path.exists():
        try:
            db = get_database(config)
            issues = db.integrity_check()
            if issues:
                for issue in issues:
                    print(f"  [X] {issue}")
            else:
                print(f"  [OK] Database at {config.database_path}")
                try:
                    stats = db.get_stats()
                    print(f"  [OK] {stats['total']} memories, {stats['archived']} archived")
                except Exception as e:
                    print(f"  [!] Could not get stats: {e}")
                    print(f"  [!] Run 'llama-memory init' to update schema")

            try:
                if db.needs_reembedding():
                    print(f"  [!] Some memories need re-embedding (model changed)")
            except:
                pass
        except Exception as e:
            print(f"  [X] Database error: {e}")
    else:
        print(f"  [!] Database not initialized. Run 'llama-memory init'")

    return 0


def cmd_serve(args):
    """Run MCP server."""
    from .mcp_server import run_server
    run_server()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Local vector memory for AI assistants",
        prog="llama-memory"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init
    p_init = subparsers.add_parser("init", help="Initialize database")
    p_init.set_defaults(func=cmd_init)

    # store
    p_store = subparsers.add_parser("store", help="Store a memory")
    p_store.add_argument("content", help="Memory content")
    p_store.add_argument("--type", "-t", default="fact",
                         choices=["fact", "decision", "event", "entity", "context", "procedure"])
    p_store.add_argument("--project", "-p", help="Project name")
    p_store.add_argument("--importance", "-i", type=int, default=5, help="1-10")
    p_store.add_argument("--tag", action="append", help="Add tag (can repeat)")
    p_store.add_argument("--summary", "-s", help="Brief summary")
    p_store.add_argument("--retention", "-r", default="long-term",
                         choices=["permanent", "long-term", "medium", "short", "session"])
    p_store.add_argument("--force", action="store_true", help="Store even if duplicate detected")
    p_store.set_defaults(func=cmd_store)

    # search
    p_search = subparsers.add_parser("search", help="Semantic search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-l", type=int, default=10)
    p_search.add_argument("--type", "-t")
    p_search.add_argument("--project", "-p")
    p_search.add_argument("--session", help="Filter by session ID")
    p_search.add_argument("--min-importance", type=int)
    p_search.add_argument("--no-hybrid", action="store_true", help="Disable hybrid ranking (use raw distance)")
    p_search.add_argument("--format", "-f", choices=["text", "json"], default="json")
    p_search.set_defaults(func=cmd_search)

    # search-text
    p_fts = subparsers.add_parser("search-text", help="Full-text search (faster)")
    p_fts.add_argument("query", help="Search query")
    p_fts.add_argument("--limit", "-l", type=int, default=10)
    p_fts.add_argument("--type", "-t")
    p_fts.add_argument("--project", "-p")
    p_fts.add_argument("--format", "-f", choices=["text", "json"], default="json")
    p_fts.set_defaults(func=cmd_search_text)

    # list
    p_list = subparsers.add_parser("list", help="List memories")
    p_list.add_argument("--limit", "-l", type=int, default=20)
    p_list.add_argument("--type", "-t")
    p_list.add_argument("--project", "-p")
    p_list.add_argument("--order-by", "-o", default="created_at",
                        choices=["created_at", "updated_at", "accessed_at", "importance"])
    p_list.add_argument("--archived", "-a", action="store_true")
    p_list.add_argument("--format", "-f", choices=["text", "json"], default="json")
    p_list.set_defaults(func=cmd_list)

    # get
    p_get = subparsers.add_parser("get", help="Get memory by ID")
    p_get.add_argument("id", type=int)
    p_get.set_defaults(func=cmd_get)

    # update
    p_update = subparsers.add_parser("update", help="Update a memory")
    p_update.add_argument("id", type=int)
    p_update.add_argument("--content", "-c")
    p_update.add_argument("--summary", "-s")
    p_update.add_argument("--importance", "-i", type=int)
    p_update.add_argument("--archive", action="store_true")
    p_update.set_defaults(func=cmd_update)

    # supersede
    p_sup = subparsers.add_parser("supersede", help="Replace a memory")
    p_sup.add_argument("old_id", type=int)
    p_sup.add_argument("content", help="New content")
    p_sup.set_defaults(func=cmd_supersede)

    # delete
    p_del = subparsers.add_parser("delete", help="Delete (archive) a memory")
    p_del.add_argument("id", type=int)
    p_del.add_argument("--hard", action="store_true", help="Permanently delete")
    p_del.set_defaults(func=cmd_delete)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show statistics")
    p_stats.set_defaults(func=cmd_stats)

    # export
    p_export = subparsers.add_parser("export", help="Export to JSON")
    p_export.add_argument("--output", "-o", help="Output file")
    p_export.add_argument("--archived", "-a", action="store_true")
    p_export.set_defaults(func=cmd_export)

    # cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Archive expired memories")
    p_cleanup.set_defaults(func=cmd_cleanup)

    # config
    p_config = subparsers.add_parser("config", help="Show configuration")
    p_config.add_argument("--show", action="store_true", help="Show current config")
    p_config.add_argument("--path", action="store_true", help="Show config file path")
    p_config.set_defaults(func=cmd_config)

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Health check")
    p_doctor.set_defaults(func=cmd_doctor)

    # serve
    p_serve = subparsers.add_parser("serve", help="Run MCP server")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
