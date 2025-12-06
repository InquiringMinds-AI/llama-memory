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


def parse_date(date_str: str) -> int:
    """Parse date string to timestamp. Supports YYYY-MM-DD or relative like '7d', '1w', '1m'."""
    from datetime import datetime, timedelta

    if not date_str:
        return None

    # Relative dates
    if date_str.endswith('d'):
        days = int(date_str[:-1])
        return int((datetime.now() - timedelta(days=days)).timestamp())
    elif date_str.endswith('w'):
        weeks = int(date_str[:-1])
        return int((datetime.now() - timedelta(weeks=weeks)).timestamp())
    elif date_str.endswith('m'):
        months = int(date_str[:-1])
        return int((datetime.now() - timedelta(days=months * 30)).timestamp())

    # Absolute date
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or relative (7d, 1w, 1m)")


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


def cmd_batch_store(args):
    """Store multiple memories from JSON file or stdin."""
    config = get_config()
    store = get_store(config)
    store.initialize()

    # Read from file or stdin
    if args.file:
        with open(args.file) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    if not isinstance(data, list):
        data = [data]

    stored = 0
    skipped = 0
    errors = []

    for i, item in enumerate(data):
        try:
            content = item.get('content')
            if not content:
                errors.append(f"Item {i}: missing content")
                continue

            memory_id, duplicate = store.store(
                content=content,
                type=item.get('type', 'fact'),
                summary=item.get('summary'),
                project=item.get('project'),
                importance=item.get('importance', 5),
                tags=item.get('tags', []),
                retention=item.get('retention', 'long-term'),
                force=args.force,
            )

            if memory_id == -1:
                skipped += 1
                if args.verbose:
                    print(f"Skipped (duplicate of #{duplicate.existing_id}): {content[:50]}...")
            else:
                stored += 1
                if args.verbose:
                    print(f"Stored #{memory_id}: {content[:50]}...")

        except Exception as e:
            errors.append(f"Item {i}: {e}")

    print(f"Stored: {stored}, Skipped: {skipped}, Errors: {len(errors)}")
    if errors and args.verbose:
        for err in errors:
            print(f"  Error: {err}")

    return 0 if not errors else 1


def cmd_import(args):
    """Import memories from a JSON export file."""
    config = get_config()
    store = get_store(config)
    store.initialize()

    with open(args.file) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Expected JSON array of memories")
        return 1

    imported = 0
    skipped = 0

    for item in data:
        # Skip archived unless --include-archived
        if item.get('archived') and not args.include_archived:
            skipped += 1
            continue

        # Skip if superseded
        if item.get('superseded_by'):
            skipped += 1
            continue

        memory_id, duplicate = store.store(
            content=item['content'],
            type=item.get('type', 'fact'),
            summary=item.get('summary'),
            project=item.get('project'),
            importance=item.get('importance', 5),
            tags=item.get('tags', []),
            retention=item.get('retention', 'long-term'),
            force=args.force,
            check_duplicates=not args.force,
        )

        if memory_id > 0:
            imported += 1
        else:
            skipped += 1

    print(f"Imported: {imported}, Skipped: {skipped}")
    return 0


def cmd_search(args):
    """Search memories."""
    config = get_config()
    store = get_store(config)

    # Parse date filters
    after = parse_date(args.after) if hasattr(args, 'after') and args.after else None
    before = parse_date(args.before) if hasattr(args, 'before') and args.before else None

    results = store.search(
        query=args.query,
        limit=args.limit,
        type=args.type,
        project=args.project,
        session_id=args.session,
        min_importance=args.min_importance,
        hybrid_ranking=not args.no_hybrid,
        after=after,
        before=before,
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

    # Parse date filters
    after = parse_date(args.after) if hasattr(args, 'after') and args.after else None
    before = parse_date(args.before) if hasattr(args, 'before') and args.before else None

    results = store.list(
        limit=args.limit,
        type=args.type,
        project=args.project,
        order_by=args.order_by,
        include_archived=args.archived,
        after=after,
        before=before,
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

    if args.format == 'json':
        print(json.dumps(stats, indent=2))
    else:
        print("=== Memory Statistics ===\n")
        print(f"Total memories: {stats['total']}")
        print(f"Archived: {stats['archived']}")
        print(f"Database size: {stats['database_size_bytes'] / 1024:.1f} KB")
        print(f"Schema version: {stats['schema_version']}")
        print(f"Embedding model: {stats['embedding_model']}")

        if stats['by_type']:
            print("\nBy type:")
            for t, count in sorted(stats['by_type'].items()):
                print(f"  {t}: {count}")

        if stats['by_project']:
            print("\nBy project:")
            for p, count in sorted(stats['by_project'].items()):
                print(f"  {p}: {count}")

        # Get most accessed memories
        conn = store.db.conn
        top_accessed = conn.execute("""
            SELECT id, content, access_count FROM memories
            WHERE archived = 0 AND access_count > 0
            ORDER BY access_count DESC LIMIT 5
        """).fetchall()

        if top_accessed:
            print("\nMost accessed:")
            for row in top_accessed:
                print(f"  [{row['id']}] ({row['access_count']}x) {row['content'][:50]}...")

        # Get oldest unaccessed memories
        old_unaccessed = conn.execute("""
            SELECT id, content, created_at FROM memories
            WHERE archived = 0 AND (accessed_at IS NULL OR access_count = 0)
            ORDER BY created_at ASC LIMIT 5
        """).fetchall()

        if old_unaccessed:
            print("\nNever accessed:")
            for row in old_unaccessed:
                print(f"  [{row['id']}] {row['content'][:50]}...")

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


def cmd_recall(args):
    """Generate context block for session start."""
    config = get_config()
    store = get_store(config)

    # Get relevant memories based on query or defaults
    if args.query:
        memories = store.search(
            query=args.query,
            limit=args.limit,
            project=args.project,
            min_importance=args.min_importance or 5,
        )
    else:
        # Get recent high-importance memories
        memories = store.list(
            limit=args.limit,
            project=args.project,
            order_by='accessed_at',
        )
        # Filter to importance >= 7 for auto-recall
        memories = [m for m in memories if m.importance >= (args.min_importance or 7)]

    if not memories:
        print("No relevant memories found.")
        return 0

    # Format as context block
    if args.format == 'markdown':
        print("# Memory Context\n")
        for m in memories:
            print(f"## [{m.type.upper()}] (importance: {m.importance})")
            if m.project:
                print(f"*Project: {m.project}*\n")
            print(m.content)
            print()
    elif args.format == 'compact':
        for m in memories:
            prefix = f"[{m.type}:{m.importance}]"
            if m.project:
                prefix += f" ({m.project})"
            print(f"{prefix} {m.content[:150]}{'...' if len(m.content) > 150 else ''}")
    else:  # json
        print(json.dumps([m.to_dict() for m in memories], indent=2))

    return 0


def cmd_related(args):
    """Find memories related to a specific memory."""
    config = get_config()
    store = get_store(config)

    # Get the source memory
    source = store.get(args.id)
    if not source:
        print(f"Memory {args.id} not found")
        return 1

    results = []

    # Find semantically similar memories
    similar = store.search(
        query=source.content,
        limit=args.limit + 1,  # +1 because source will match itself
    )
    for m in similar:
        if m.id != source.id:
            m.tags = m.tags + ['similar']
            results.append(m)

    # Find memories from same session
    if source.session_id:
        session_memories = store.search(
            query=source.content,
            limit=50,
            session_id=source.session_id,
        )
        for m in session_memories:
            if m.id != source.id and m.id not in [r.id for r in results]:
                m.tags = m.tags + ['same-session']
                results.append(m)

    # Find memories that superseded or were superseded by this one
    if source.superseded_by:
        newer = store.get(source.superseded_by)
        if newer:
            newer.tags = newer.tags + ['supersedes-this']
            results.append(newer)

    # Check if this supersedes something
    conn = store.db.conn
    older_row = conn.execute(
        "SELECT id FROM memories WHERE superseded_by = ?", (source.id,)
    ).fetchone()
    if older_row:
        older = store.get(older_row['id'])
        if older:
            older.tags = older.tags + ['superseded-by-this']
            results.append(older)

    # Deduplicate and limit
    seen = set()
    unique_results = []
    for m in results:
        if m.id not in seen:
            seen.add(m.id)
            unique_results.append(m)

    unique_results = unique_results[:args.limit]

    if args.format == 'json':
        print(json.dumps([m.to_dict() for m in unique_results], indent=2))
    else:
        print(f"Related to [{source.id}]: {source.content[:80]}...\n")
        for m in unique_results:
            relation = ', '.join([t for t in m.tags if t in ['similar', 'same-session', 'supersedes-this', 'superseded-by-this']])
            print(f"[{m.id}] ({relation}) {m.content[:100]}{'...' if len(m.content) > 100 else ''}")

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


def cmd_version(args):
    """Show version information."""
    from . import __version__
    config = get_config()
    print(f"llama-memory {__version__}")
    print(f"Embedding model: {config.embedding_model_version}")
    print(f"Database: {config.database_path}")
    return 0


def cmd_backup(args):
    """Create a backup of the database."""
    import shutil
    from datetime import datetime

    config = get_config()

    if not config.database_path.exists():
        print("No database to backup")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"memory_backup_{timestamp}.db"

    if args.output:
        backup_path = Path(args.output)
    else:
        backup_path = config.database_path.parent / backup_name

    shutil.copy2(config.database_path, backup_path)
    size_kb = backup_path.stat().st_size / 1024
    print(f"Backed up to {backup_path} ({size_kb:.1f} KB)")
    return 0


def cmd_tags(args):
    """List all tags in use."""
    config = get_config()
    store = get_store(config)

    conn = store.db.conn
    rows = conn.execute("""
        SELECT tags FROM memories
        WHERE archived = 0 AND tags IS NOT NULL AND tags != '[]'
    """).fetchall()

    tag_counts = {}
    for row in rows:
        tags = json.loads(row['tags'])
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if not tag_counts:
        print("No tags found")
        return 0

    if args.format == 'json':
        print(json.dumps(tag_counts, indent=2))
    else:
        print("Tags in use:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            print(f"  {tag}: {count}")

    return 0


def cmd_restore(args):
    """Restore database from a backup."""
    import shutil

    config = get_config()
    backup_path = Path(args.file)

    if not backup_path.exists():
        print(f"Backup file not found: {backup_path}")
        return 1

    # Safety check - create a backup of current before restoring
    if config.database_path.exists() and not args.no_backup:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safety_backup = config.database_path.parent / f"memory_pre_restore_{timestamp}.db"
        shutil.copy2(config.database_path, safety_backup)
        print(f"Current database backed up to: {safety_backup}")

    shutil.copy2(backup_path, config.database_path)
    size_kb = config.database_path.stat().st_size / 1024
    print(f"Restored from {backup_path} ({size_kb:.1f} KB)")
    return 0


def cmd_projects(args):
    """List all projects in use."""
    config = get_config()
    store = get_store(config)

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

    if not rows:
        print("No projects found")
        return 0

    if args.format == 'json':
        projects = {
            row['project']: {
                'count': row['count'],
                'max_importance': row['max_importance'],
                'latest': row['latest']
            }
            for row in rows
        }
        print(json.dumps(projects, indent=2))
    else:
        print("Projects:")
        for row in rows:
            from datetime import datetime
            latest = datetime.fromtimestamp(row['latest']).strftime("%Y-%m-%d")
            print(f"  {row['project']}: {row['count']} memories (latest: {latest})")

    return 0


def cmd_unarchive(args):
    """Restore an archived memory."""
    config = get_config()
    store = get_store(config)

    success = store.unarchive(args.id)
    if success:
        print(f"Unarchived memory {args.id}")
    else:
        print(f"Memory {args.id} not found")
        return 1
    return 0


def cmd_reembed(args):
    """Regenerate embeddings for memories."""
    config = get_config()
    store = get_store(config)

    if args.id:
        count = store.reembed(id=args.id)
        if count:
            print(f"Re-embedded memory {args.id}")
        else:
            print(f"Memory {args.id} not found")
            return 1
    else:
        # Show what would be re-embedded
        conn = store.db.conn
        if args.old_model:
            rows = conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE archived = 0 AND embedding_model = ?",
                (args.old_model,)
            ).fetchone()
            target = f"memories with model '{args.old_model}'"
        else:
            rows = conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE archived = 0").fetchone()
            target = "all memories"

        if not args.yes:
            print(f"This will re-embed {rows['cnt']} {target}")
            print(f"New model: {config.embedding_model_version}")
            print("This may take a while. Use --yes to confirm.")
            return 0

        print(f"Re-embedding {rows['cnt']} {target}...")
        count = store.reembed(model_filter=args.old_model)
        print(f"Re-embedded {count} memories")

    return 0


def cmd_sessions(args):
    """List and browse sessions."""
    config = get_config()
    store = get_store(config)

    conn = store.db.conn

    if args.session_id:
        # Show memories from a specific session
        rows = conn.execute("""
            SELECT id, content, type, importance, created_at
            FROM memories
            WHERE session_id = ? AND archived = 0
            ORDER BY created_at ASC
        """, (args.session_id,)).fetchall()

        if not rows:
            print(f"No memories found for session: {args.session_id}")
            return 1

        if args.format == 'json':
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
            print(json.dumps(memories, indent=2))
        else:
            print(f"Session: {args.session_id} ({len(rows)} memories)\n")
            for row in rows:
                from datetime import datetime
                ts = datetime.fromtimestamp(row['created_at']).strftime("%H:%M:%S")
                print(f"  [{row['id']}] {ts} ({row['type']}) {row['content'][:80]}...")
    else:
        # List all sessions
        rows = conn.execute("""
            SELECT session_id, COUNT(*) as count,
                   MIN(created_at) as started,
                   MAX(created_at) as ended
            FROM memories
            WHERE session_id IS NOT NULL AND archived = 0
            GROUP BY session_id
            ORDER BY started DESC
            LIMIT ?
        """, (args.limit,)).fetchall()

        if not rows:
            print("No sessions found")
            return 0

        if args.format == 'json':
            sessions = [
                {
                    'session_id': row['session_id'],
                    'count': row['count'],
                    'started': row['started'],
                    'ended': row['ended']
                }
                for row in rows
            ]
            print(json.dumps(sessions, indent=2))
        else:
            print("Recent sessions:")
            for row in rows:
                from datetime import datetime
                started = datetime.fromtimestamp(row['started']).strftime("%Y-%m-%d %H:%M")
                duration_mins = (row['ended'] - row['started']) / 60
                print(f"  {row['session_id']}")
                print(f"    {row['count']} memories, started {started}, ~{duration_mins:.0f}min")

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

    # batch-store
    p_batch = subparsers.add_parser("batch-store", help="Store multiple memories from JSON")
    p_batch.add_argument("--file", "-f", help="JSON file (or stdin if not provided)")
    p_batch.add_argument("--force", action="store_true", help="Store even if duplicates detected")
    p_batch.add_argument("--verbose", "-v", action="store_true", help="Show each item stored")
    p_batch.set_defaults(func=cmd_batch_store)

    # import
    p_import = subparsers.add_parser("import", help="Import memories from export JSON")
    p_import.add_argument("file", help="JSON file from 'export' command")
    p_import.add_argument("--force", action="store_true", help="Skip duplicate detection")
    p_import.add_argument("--include-archived", action="store_true", help="Also import archived memories")
    p_import.set_defaults(func=cmd_import)

    # search
    p_search = subparsers.add_parser("search", help="Semantic search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-l", type=int, default=10)
    p_search.add_argument("--type", "-t")
    p_search.add_argument("--project", "-p")
    p_search.add_argument("--session", help="Filter by session ID")
    p_search.add_argument("--min-importance", type=int)
    p_search.add_argument("--after", help="Only memories after date (YYYY-MM-DD or 7d, 1w, 1m)")
    p_search.add_argument("--before", help="Only memories before date (YYYY-MM-DD or 7d, 1w, 1m)")
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
    p_list.add_argument("--after", help="Only memories after date (YYYY-MM-DD or 7d, 1w, 1m)")
    p_list.add_argument("--before", help="Only memories before date (YYYY-MM-DD or 7d, 1w, 1m)")
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
    p_stats.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_stats.set_defaults(func=cmd_stats)

    # export
    p_export = subparsers.add_parser("export", help="Export to JSON")
    p_export.add_argument("--output", "-o", help="Output file")
    p_export.add_argument("--archived", "-a", action="store_true")
    p_export.set_defaults(func=cmd_export)

    # cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Archive expired memories")
    p_cleanup.set_defaults(func=cmd_cleanup)

    # recall
    p_recall = subparsers.add_parser("recall", help="Generate context block for session start")
    p_recall.add_argument("query", nargs="?", help="Optional search query")
    p_recall.add_argument("--limit", "-l", type=int, default=10)
    p_recall.add_argument("--project", "-p", help="Filter by project")
    p_recall.add_argument("--min-importance", "-i", type=int, help="Minimum importance (default 7 for auto, 5 with query)")
    p_recall.add_argument("--format", "-f", choices=["markdown", "compact", "json"], default="compact")
    p_recall.set_defaults(func=cmd_recall)

    # related
    p_related = subparsers.add_parser("related", help="Find memories related to a specific memory")
    p_related.add_argument("id", type=int, help="Memory ID to find relations for")
    p_related.add_argument("--limit", "-l", type=int, default=10)
    p_related.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_related.set_defaults(func=cmd_related)

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

    # version
    p_version = subparsers.add_parser("version", help="Show version info")
    p_version.set_defaults(func=cmd_version)

    # backup
    p_backup = subparsers.add_parser("backup", help="Create database backup")
    p_backup.add_argument("--output", "-o", help="Output path (default: timestamped in data dir)")
    p_backup.set_defaults(func=cmd_backup)

    # tags
    p_tags = subparsers.add_parser("tags", help="List all tags in use")
    p_tags.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_tags.set_defaults(func=cmd_tags)

    # restore
    p_restore = subparsers.add_parser("restore", help="Restore database from backup")
    p_restore.add_argument("file", help="Backup file to restore from")
    p_restore.add_argument("--no-backup", action="store_true", help="Don't backup current database before restore")
    p_restore.set_defaults(func=cmd_restore)

    # projects
    p_projects = subparsers.add_parser("projects", help="List all projects")
    p_projects.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_projects.set_defaults(func=cmd_projects)

    # sessions
    p_sessions = subparsers.add_parser("sessions", help="List and browse sessions")
    p_sessions.add_argument("session_id", nargs="?", help="View specific session")
    p_sessions.add_argument("--limit", "-l", type=int, default=20, help="Number of sessions to show")
    p_sessions.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_sessions.set_defaults(func=cmd_sessions)

    # unarchive
    p_unarchive = subparsers.add_parser("unarchive", help="Restore an archived memory")
    p_unarchive.add_argument("id", type=int, help="Memory ID to unarchive")
    p_unarchive.set_defaults(func=cmd_unarchive)

    # reembed
    p_reembed = subparsers.add_parser("reembed", help="Regenerate embeddings for memories")
    p_reembed.add_argument("--id", type=int, help="Specific memory ID to re-embed")
    p_reembed.add_argument("--old-model", help="Only re-embed memories with this embedding model version")
    p_reembed.add_argument("--yes", "-y", action="store_true", help="Confirm re-embedding all memories")
    p_reembed.set_defaults(func=cmd_reembed)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
