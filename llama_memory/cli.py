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

    cascade = getattr(args, 'cascade', False)
    store.delete(args.id, hard=args.hard, cascade=cascade)
    action = "Deleted" if args.hard else "Archived"
    cascade_note = " (with children)" if cascade else ""
    print(f"{action} memory {args.id}{cascade_note}")
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


def cmd_vacuum(args):
    """Optimize the database by running VACUUM."""
    config = get_config()

    if not config.database_path.exists():
        print("No database to vacuum")
        return 1

    # Get size before
    size_before = config.database_path.stat().st_size

    # Run vacuum
    import sqlite3
    conn = sqlite3.connect(config.database_path)
    conn.execute("VACUUM")
    conn.close()

    # Get size after
    size_after = config.database_path.stat().st_size
    saved = size_before - size_after

    print(f"Database optimized")
    print(f"  Before: {size_before / 1024:.1f} KB")
    print(f"  After:  {size_after / 1024:.1f} KB")
    if saved > 0:
        print(f"  Saved:  {saved / 1024:.1f} KB ({saved * 100 / size_before:.1f}%)")
    else:
        print(f"  No space reclaimed (database already optimized)")

    return 0


def cmd_history(args):
    """Show history for a specific memory."""
    config = get_config()
    store = get_store(config)

    conn = store.db.conn

    # Get memory info
    memory = store.get(args.id)
    if not memory:
        print(f"Memory {args.id} not found")
        return 1

    # Get log entries
    rows = conn.execute("""
        SELECT action, timestamp, details
        FROM memory_log
        WHERE memory_id = ?
        ORDER BY timestamp ASC
    """, (args.id,)).fetchall()

    if args.format == 'json':
        history = {
            'memory': memory.to_dict(),
            'events': [
                {
                    'action': row['action'],
                    'timestamp': row['timestamp'],
                    'details': json.loads(row['details']) if row['details'] else None
                }
                for row in rows
            ]
        }
        print(json.dumps(history, indent=2))
    else:
        from datetime import datetime
        print(f"History for memory {args.id}:")
        print(f"  Content: {memory.content[:80]}...")
        print(f"  Created: {datetime.fromtimestamp(memory.created_at).strftime('%Y-%m-%d %H:%M')}")
        print(f"  Accessed: {memory.access_count}x")
        print()
        print("Events:")
        for row in rows:
            ts = datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            details = ""
            if row['details']:
                d = json.loads(row['details'])
                details = f" - {d}" if d else ""
            print(f"  [{ts}] {row['action']}{details}")

    return 0


def cmd_count(args):
    """Count memories with optional filters."""
    config = get_config()
    store = get_store(config)

    conn = store.db.conn

    sql = "SELECT COUNT(*) as cnt FROM memories WHERE 1=1"
    params = []

    if not args.include_archived:
        sql += " AND archived = 0"

    if args.type:
        sql += " AND type = ?"
        params.append(args.type)

    if args.project:
        sql += " AND project = ?"
        params.append(args.project)

    if args.min_importance:
        sql += " AND importance >= ?"
        params.append(args.min_importance)

    row = conn.execute(sql, params).fetchone()
    count = row['cnt']

    if args.quiet:
        print(count)
    else:
        filters = []
        if args.type:
            filters.append(f"type={args.type}")
        if args.project:
            filters.append(f"project={args.project}")
        if args.min_importance:
            filters.append(f"importance>={args.min_importance}")
        if args.include_archived:
            filters.append("including archived")

        filter_str = f" ({', '.join(filters)})" if filters else ""
        print(f"{count} memories{filter_str}")

    return 0


def cmd_link(args):
    """Link two memories together."""
    config = get_config()
    store = get_store(config)

    try:
        created = store.link(
            source_id=args.source,
            target_id=args.target,
            link_type=args.type,
            note=args.note,
            weight=args.weight,
        )
        if created:
            weight_str = f" weight={args.weight}" if args.weight != 1.0 else ""
            print(f"Linked memory {args.source} -> {args.target} ({args.type}{weight_str})")
        else:
            print(f"Link already exists between {args.source} and {args.target}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_unlink(args):
    """Remove a link between two memories."""
    config = get_config()
    store = get_store(config)

    removed = store.unlink(args.source, args.target)
    if removed:
        print(f"Removed link between {args.source} and {args.target}")
    else:
        print(f"No link found between {args.source} and {args.target}")
        return 1

    return 0


def cmd_links(args):
    """Show links for a memory."""
    config = get_config()
    store = get_store(config)

    memory = store.get(args.id)
    if not memory:
        print(f"Memory {args.id} not found")
        return 1

    links = store.get_links(args.id)

    if args.format == 'json':
        print(json.dumps(links, indent=2))
    else:
        if not links:
            print(f"No links for memory {args.id}")
        else:
            print(f"Links for memory {args.id}: {memory.content[:60]}...\n")
            for link in links:
                direction = "->" if link['direction'] == 'outgoing' else "<-"
                note = f" ({link['note']})" if link['note'] else ""
                weight_str = f" [{link['weight']:.1f}]" if link.get('weight', 1.0) != 1.0 else ""
                print(f"  {direction} [{link['linked_id']}] {link['link_type']}{weight_str}{note}")
                print(f"     {link['content'][:70]}...")

    return 0


def cmd_types(args):
    """List all memory types in use."""
    config = get_config()
    store = get_store(config)

    conn = store.db.conn
    rows = conn.execute("""
        SELECT type, COUNT(*) as count
        FROM memories
        WHERE archived = 0
        GROUP BY type
        ORDER BY count DESC
    """).fetchall()

    if not rows:
        print("No memories found")
        return 0

    if args.format == 'json':
        types = {row['type']: row['count'] for row in rows}
        print(json.dumps(types, indent=2))
    else:
        print("Memory types:")
        for row in rows:
            print(f"  {row['type']}: {row['count']}")

    return 0


# ========== v2.0 Commands ==========

def cmd_topics(args):
    """List or manage topics."""
    config = get_config()
    store = get_store(config)

    if args.set:
        # Set topic for a memory
        memory_id = int(args.set[0])
        topic = args.set[1]
        if store.set_topic(memory_id, topic):
            print(f"Set topic '{topic}' for memory {memory_id}")
        else:
            print(f"Memory {memory_id} not found")
            return 1
    elif args.get:
        # Get memories by topic
        mems = store.get_by_topic(args.get, limit=args.limit)
        if args.format == 'json':
            print(json.dumps([m.to_dict() for m in mems], indent=2))
        else:
            print(f"Memories with topic '{args.get}':")
            for m in mems:
                print(f"  [{m.id}] {m.content[:80]}...")
    else:
        # List all topics
        topics = store.get_topics()
        if args.format == 'json':
            print(json.dumps(topics, indent=2))
        else:
            if not topics:
                print("No topics found")
            else:
                print("Topics:")
                for t in topics:
                    print(f"  {t['topic']}: {t['count']} memories")

    return 0


def cmd_merge(args):
    """Merge multiple memories into one."""
    config = get_config()
    store = get_store(config)

    try:
        merged_id = store.merge(
            source_ids=args.ids,
            merged_content=args.content,
            archive_sources=args.archive,
        )
        print(f"Created merged memory {merged_id} from {args.ids}")
        if args.archive:
            print("Source memories archived")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_children(args):
    """List child memories of a parent."""
    config = get_config()
    store = get_store(config)

    children = store.get_children(args.id)

    if args.format == 'json':
        print(json.dumps([c.to_dict() for c in children], indent=2))
    else:
        if not children:
            print(f"No children for memory {args.id}")
        else:
            print(f"Children of memory {args.id}:")
            for c in children:
                print(f"  [{c.id}] ({c.type}) {c.content[:80]}...")

    return 0


def cmd_parent(args):
    """Set or show parent of a memory."""
    config = get_config()
    store = get_store(config)

    if args.set is not None:
        parent_id = args.set if args.set > 0 else None
        try:
            if store.set_parent(args.id, parent_id):
                if parent_id:
                    print(f"Set parent of memory {args.id} to {parent_id}")
                else:
                    print(f"Removed parent from memory {args.id}")
            else:
                print(f"Memory {args.id} not found")
                return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        # Show ancestors
        ancestors = store.get_ancestors(args.id)
        if not ancestors:
            print(f"Memory {args.id} has no parent")
        else:
            print(f"Ancestors of memory {args.id}:")
            for i, a in enumerate(ancestors):
                indent = "  " * (i + 1)
                print(f"{indent}[{a.id}] {a.content[:60]}...")

    return 0


def cmd_confidence(args):
    """Set confidence level for a memory."""
    config = get_config()
    store = get_store(config)

    try:
        if store.set_confidence(args.id, args.value):
            print(f"Set confidence of memory {args.id} to {args.value:.0%}")
        else:
            print(f"Memory {args.id} not found")
            return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_conflicts(args):
    """Check or list memory conflicts."""
    config = get_config()
    store = get_store(config)

    if args.check:
        # Check conflicts for a specific memory
        conflicts = store.check_conflicts(args.check, threshold=args.threshold)
        if args.format == 'json':
            print(json.dumps(conflicts, indent=2))
        else:
            if not conflicts:
                print(f"No potential conflicts found for memory {args.check}")
            else:
                print(f"Potential conflicts for memory {args.check}:")
                for c in conflicts:
                    print(f"  [{c['memory_id']}] similarity={c['similarity']:.1%}")
                    print(f"    {c['content'][:100]}...")
    elif args.resolve:
        # Resolve a conflict
        if store.resolve_conflict(args.resolve[0], args.resolve[1], note=args.note):
            print(f"Resolved conflict {args.resolve[0]} with memory {args.resolve[1]}")
        else:
            print(f"Conflict {args.resolve[0]} not found")
            return 1
    else:
        # List all conflicts
        conflicts = store.get_conflicts(conflict_type=args.type)
        if args.format == 'json':
            print(json.dumps(conflicts, indent=2))
        else:
            if not conflicts:
                print("No conflicts found")
            else:
                print("Memory conflicts:")
                for c in conflicts:
                    status = f"[{c['conflict_type']}]"
                    print(f"  {status} {c['memory1_id']} <-> {c['memory2_id']} (sim={c['similarity']:.1%})")

    return 0


def cmd_search_history(args):
    """Show search history."""
    config = get_config()
    store = get_store(config)

    if args.popular:
        queries = store.get_popular_queries(limit=args.limit)
        if args.format == 'json':
            print(json.dumps(queries, indent=2))
        else:
            print("Popular queries:")
            for q in queries:
                print(f"  {q['count']}x: {q['query']}")
    else:
        history = store.get_search_history(limit=args.limit, session_id=args.session)
        if args.format == 'json':
            print(json.dumps(history, indent=2))
        else:
            print("Search history:")
            for h in history:
                from datetime import datetime
                ts = datetime.fromtimestamp(h['created_at']).strftime('%Y-%m-%d %H:%M')
                print(f"  [{ts}] {h['query']} ({h['result_count']} results)")

    return 0


def cmd_export_md(args):
    """Export memories as Markdown."""
    config = get_config()
    store = get_store(config)

    md = store.export_markdown(include_archived=args.archived)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(md)
        print(f"Exported to {args.output}")
    else:
        print(md)

    return 0


def cmd_export_csv(args):
    """Export memories as CSV."""
    config = get_config()
    store = get_store(config)

    csv_out = store.export_csv(include_archived=args.archived)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(csv_out)
        print(f"Exported to {args.output}")
    else:
        print(csv_out)

    return 0


# ========== v2.1 Decay Commands ==========

def cmd_decay(args):
    """Run decay process or check status."""
    config = get_config()
    store = get_store(config)

    from llama_memory.database import DECAY_START_DAYS, DECAY_ARCHIVE_DAYS

    if args.status:
        # Show decay status for a specific memory
        status = store.get_decay_status(args.status)
        if args.format == 'json':
            print(json.dumps(status, indent=2))
        else:
            if 'error' in status:
                print(f"Error: {status['error']}")
                return 1
            print(f"Decay status for memory {args.status}:")
            print(f"  Protected: {'Yes' if status['protected'] else 'No'}")
            if status['protection_reasons']:
                print(f"  Reasons: {', '.join(status['protection_reasons'])}")
            print(f"  Days since access: {status['days_since_access']}")
            print(f"  Decay starts at: {DECAY_START_DAYS} days")
            print(f"  Auto-archive at: {DECAY_ARCHIVE_DAYS} days")
            if status['in_decay_window']:
                print(f"  Status: IN DECAY WINDOW (ranking reduced)")
            elif status['will_decay']:
                print(f"  Status: WILL BE ARCHIVED on next decay run")
            elif status['protected']:
                print(f"  Status: Protected from decay")
            else:
                print(f"  Status: Safe (accessed recently)")
    elif args.run:
        # Run decay process
        result = store.run_decay(dry_run=args.dry_run)
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            if args.dry_run:
                print(f"DRY RUN - Would archive {len(result['candidates'])} memories:")
            else:
                print(f"Archived {result['archived']} memories:")
            for c in result['candidates']:
                from datetime import datetime
                last = datetime.fromtimestamp(c['last_access']).strftime('%Y-%m-%d')
                print(f"  [{c['id']}] ({c['type']}) last accessed {last}")
                print(f"    {c['content']}...")
    else:
        # Show decay overview
        last_run = store.get_last_decay_run()
        candidates = store.get_decay_candidates()

        if args.format == 'json':
            print(json.dumps({
                'last_run': last_run,
                'decay_start_days': DECAY_START_DAYS,
                'decay_archive_days': DECAY_ARCHIVE_DAYS,
                'candidates_count': len(candidates),
            }, indent=2))
        else:
            print("Decay System Status:")
            print(f"  Decay starts after: {DECAY_START_DAYS} days without access")
            print(f"  Auto-archive after: {DECAY_ARCHIVE_DAYS} days without access")
            if last_run:
                from datetime import datetime
                print(f"  Last decay run: {datetime.fromtimestamp(last_run).strftime('%Y-%m-%d %H:%M')}")
            else:
                print("  Last decay run: Never")
            print(f"  Candidates for archive: {len(candidates)}")
            if candidates:
                print("\n  Run 'llama-memory decay --run --dry-run' to preview")
                print("  Run 'llama-memory decay --run' to archive them")

    return 0


def cmd_protect(args):
    """Protect or unprotect a memory from decay."""
    config = get_config()
    store = get_store(config)

    protected = not args.remove
    if store.set_protected(args.id, protected):
        if protected:
            print(f"Memory {args.id} is now protected from decay")
        else:
            print(f"Memory {args.id} protection removed (may decay if not otherwise protected)")
    else:
        print(f"Memory {args.id} not found")
        return 1

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


# ========== v2.7 Session Persistence Commands ==========

def cmd_session_save(args):
    """Save current session state."""
    from .session import SessionStore
    config = get_config()
    store = get_store(config)
    session_store = SessionStore(store)

    # Parse progress if provided as JSON string
    progress = None
    if args.progress:
        try:
            progress = json.loads(args.progress)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON for progress: {args.progress}")
            return 1

    # Parse list arguments
    files = args.file or []
    decisions = args.decision or []
    next_steps = args.next or []
    blockers = args.blocker or []
    tags = args.tag or []

    session = session_store.save(
        title=args.title,
        summary=args.summary,
        task_description=args.task,
        progress=progress,
        files_touched=files,
        decisions=decisions,
        next_steps=next_steps,
        blockers=blockers,
        project=args.project,
        notes=args.notes,
        tags=tags,
    )

    if args.format == 'json':
        print(json.dumps({
            "id": session.id,
            "title": session.title,
            "summary": session.summary,
            "project": session.project,
        }, indent=2))
    else:
        print(f"Session saved: ID {session.id}")
        print(f"  Title: {session.title}")
        if session.project:
            print(f"  Project: {session.project}")
        if session.next_steps:
            print(f"  Next steps: {len(session.next_steps)} items")

    return 0


def cmd_session_resume(args):
    """Get resume prompt for a session."""
    from .session import SessionStore
    config = get_config()
    store = get_store(config)
    session_store = SessionStore(store)

    resume_prompt = session_store.resume(
        session_id=args.id,
        project=args.project,
    )

    if resume_prompt:
        print(resume_prompt)
    else:
        print("No session found to resume.")
        return 1

    return 0


def cmd_session_list(args):
    """List saved sessions."""
    from .session import SessionStore
    config = get_config()
    store = get_store(config)
    session_store = SessionStore(store)

    sessions = session_store.list(
        project=args.project,
        limit=args.limit,
    )

    if not sessions:
        print("No sessions found.")
        return 0

    if args.format == 'json':
        result = []
        for s in sessions:
            result.append({
                "id": s.id,
                "title": s.title,
                "summary": s.summary,
                "project": s.project,
                "ended_at": s.ended_at,
                "next_steps": s.next_steps,
            })
        print(json.dumps(result, indent=2))
    else:
        print("Saved sessions:")
        for s in sessions:
            from datetime import datetime
            ended = datetime.fromtimestamp(s.ended_at).strftime("%Y-%m-%d %H:%M") if s.ended_at else "?"
            project_str = f" [{s.project}]" if s.project else ""
            print(f"\n  [{s.id}] {s.title}{project_str}")
            print(f"      Saved: {ended}")
            if s.next_steps:
                print(f"      Next: {s.next_steps[0]}" + (f" (+{len(s.next_steps)-1} more)" if len(s.next_steps) > 1 else ""))

    return 0


def cmd_session_get(args):
    """Get details of a session."""
    from .session import SessionStore
    config = get_config()
    store = get_store(config)
    session_store = SessionStore(store)

    session = session_store.get(args.id)

    if not session:
        print(f"Session {args.id} not found.")
        return 1

    if args.format == 'json':
        print(json.dumps({
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
        }, indent=2))
    else:
        print(f"Session {session.id}: {session.title}")
        if session.project:
            print(f"Project: {session.project}")
        if session.summary:
            print(f"\nSummary: {session.summary}")
        if session.task_description:
            print(f"\nTask: {session.task_description}")
        if session.progress:
            c = session.progress.get('completed', 0)
            t = session.progress.get('total', 0)
            print(f"\nProgress: {c}/{t}")
        if session.files_touched:
            print(f"\nFiles: {', '.join(session.files_touched)}")
        if session.decisions:
            print("\nDecisions:")
            for d in session.decisions:
                print(f"  - {d}")
        if session.next_steps:
            print("\nNext steps:")
            for n in session.next_steps:
                print(f"  - {n}")
        if session.blockers:
            print("\nBlockers:")
            for b in session.blockers:
                print(f"  - {b}")
        if session.notes:
            print(f"\nNotes: {session.notes}")

    return 0


# ========== Hooks Commands ==========

def cmd_hooks_install(args):
    """Install Claude Code hooks for auto session capture."""
    import json
    from pathlib import Path
    from .hooks import AUTO_SESSION_SAVE, AUTO_SESSION_RESUME

    # Determine settings file location
    settings_path = Path.home() / ".claude" / "settings.json"

    # Create .claude directory if needed
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing settings or create new
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                settings = {}
    else:
        settings = {}

    # Ensure hooks structure exists
    if 'hooks' not in settings:
        settings['hooks'] = {}

    # Hook configurations
    save_hook = {
        "hooks": [{
            "type": "command",
            "command": f"python3 {AUTO_SESSION_SAVE}",
            "timeout": 30
        }]
    }

    resume_hook = {
        "hooks": [{
            "type": "command",
            "command": f"python3 {AUTO_SESSION_RESUME}",
            "timeout": 10
        }]
    }

    # Install PreCompact hook
    if 'PreCompact' not in settings['hooks']:
        settings['hooks']['PreCompact'] = []
    # Check if our hook already exists
    save_cmd = f"python3 {AUTO_SESSION_SAVE}"
    existing_precompact = [h for h in settings['hooks']['PreCompact']
                          if any(save_cmd in hk.get('command', '') for hk in h.get('hooks', []))]
    if not existing_precompact:
        settings['hooks']['PreCompact'].append(save_hook)
        print("Installed: PreCompact hook (auto-save before compaction)")
    else:
        print("Already installed: PreCompact hook")

    # Install SessionEnd hook
    if 'SessionEnd' not in settings['hooks']:
        settings['hooks']['SessionEnd'] = []
    existing_sessionend = [h for h in settings['hooks']['SessionEnd']
                          if any(save_cmd in hk.get('command', '') for hk in h.get('hooks', []))]
    if not existing_sessionend:
        settings['hooks']['SessionEnd'].append(save_hook)
        print("Installed: SessionEnd hook (auto-save on exit)")
    else:
        print("Already installed: SessionEnd hook")

    # Install SessionStart hook
    if 'SessionStart' not in settings['hooks']:
        settings['hooks']['SessionStart'] = []
    resume_cmd = f"python3 {AUTO_SESSION_RESUME}"
    existing_sessionstart = [h for h in settings['hooks']['SessionStart']
                            if any(resume_cmd in hk.get('command', '') for hk in h.get('hooks', []))]
    if not existing_sessionstart:
        settings['hooks']['SessionStart'].append(resume_hook)
        print("Installed: SessionStart hook (auto-resume context)")
    else:
        print("Already installed: SessionStart hook")

    # Write settings
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)

    print(f"\nHooks configuration saved to: {settings_path}")
    print("\nRestart Claude Code to activate hooks.")

    return 0


def cmd_hooks_status(args):
    """Check if hooks are installed."""
    import json
    from pathlib import Path
    from .hooks import AUTO_SESSION_SAVE, AUTO_SESSION_RESUME

    settings_path = Path.home() / ".claude" / "settings.json"

    if not settings_path.exists():
        print("No Claude Code settings found.")
        print(f"Expected at: {settings_path}")
        print("\nRun 'mem hooks-install' to install hooks.")
        return 1

    with open(settings_path, 'r') as f:
        try:
            settings = json.load(f)
        except json.JSONDecodeError:
            print("Invalid settings.json")
            return 1

    hooks = settings.get('hooks', {})

    print("Claude Code Hooks Status")
    print("=" * 40)

    # Check each hook
    save_cmd = f"python3 {AUTO_SESSION_SAVE}"
    resume_cmd = f"python3 {AUTO_SESSION_RESUME}"

    def check_hook(hook_name, cmd):
        hook_list = hooks.get(hook_name, [])
        for h in hook_list:
            for hk in h.get('hooks', []):
                if cmd in hk.get('command', ''):
                    return True
        return False

    precompact_ok = check_hook('PreCompact', save_cmd)
    sessionend_ok = check_hook('SessionEnd', save_cmd)
    sessionstart_ok = check_hook('SessionStart', resume_cmd)

    print(f"PreCompact (auto-save):    {'Installed' if precompact_ok else 'Not installed'}")
    print(f"SessionEnd (auto-save):    {'Installed' if sessionend_ok else 'Not installed'}")
    print(f"SessionStart (auto-resume): {'Installed' if sessionstart_ok else 'Not installed'}")

    print(f"\nSettings file: {settings_path}")

    if not all([precompact_ok, sessionend_ok, sessionstart_ok]):
        print("\nRun 'mem hooks-install' to install missing hooks.")
        return 1

    return 0


def cmd_hooks_uninstall(args):
    """Remove Claude Code hooks."""
    import json
    from pathlib import Path
    from .hooks import AUTO_SESSION_SAVE, AUTO_SESSION_RESUME

    settings_path = Path.home() / ".claude" / "settings.json"

    if not settings_path.exists():
        print("No Claude Code settings found.")
        return 0

    with open(settings_path, 'r') as f:
        try:
            settings = json.load(f)
        except json.JSONDecodeError:
            print("Invalid settings.json")
            return 1

    hooks = settings.get('hooks', {})
    save_cmd = f"python3 {AUTO_SESSION_SAVE}"
    resume_cmd = f"python3 {AUTO_SESSION_RESUME}"

    removed = 0

    for hook_name in ['PreCompact', 'SessionEnd', 'SessionStart']:
        if hook_name in hooks:
            original_len = len(hooks[hook_name])
            # Filter out our hooks
            if hook_name == 'SessionStart':
                cmd_to_remove = resume_cmd
            else:
                cmd_to_remove = save_cmd

            hooks[hook_name] = [
                h for h in hooks[hook_name]
                if not any(cmd_to_remove in hk.get('command', '') for hk in h.get('hooks', []))
            ]

            if len(hooks[hook_name]) < original_len:
                removed += 1
                print(f"Removed: {hook_name} hook")

            # Clean up empty hook arrays
            if not hooks[hook_name]:
                del hooks[hook_name]

    if removed > 0:
        settings['hooks'] = hooks
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"\nRemoved {removed} hook(s).")
        print("Restart Claude Code to apply changes.")
    else:
        print("No llama-memory hooks found to remove.")

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
    p_del.add_argument("--cascade", action="store_true", help="Also delete child memories recursively")
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

    # vacuum
    p_vacuum = subparsers.add_parser("vacuum", help="Optimize database (reclaim space)")
    p_vacuum.set_defaults(func=cmd_vacuum)

    # history
    p_history = subparsers.add_parser("history", help="Show history for a memory")
    p_history.add_argument("id", type=int, help="Memory ID")
    p_history.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_history.set_defaults(func=cmd_history)

    # count
    p_count = subparsers.add_parser("count", help="Count memories")
    p_count.add_argument("--type", "-t", help="Filter by type")
    p_count.add_argument("--project", "-p", help="Filter by project")
    p_count.add_argument("--min-importance", "-i", type=int, help="Minimum importance")
    p_count.add_argument("--include-archived", "-a", action="store_true", help="Include archived")
    p_count.add_argument("--quiet", "-q", action="store_true", help="Output number only")
    p_count.set_defaults(func=cmd_count)

    # link
    p_link = subparsers.add_parser("link", help="Link two memories together")
    p_link.add_argument("source", type=int, help="Source memory ID")
    p_link.add_argument("target", type=int, help="Target memory ID")
    p_link.add_argument("--type", "-t", default="related",
                        help="Link type (related, depends_on, contradicts, etc.)")
    p_link.add_argument("--note", "-n", help="Note about the relationship")
    p_link.add_argument("--weight", "-w", type=float, default=1.0,
                        help="Relationship strength 0.0-1.0 (default: 1.0)")
    p_link.set_defaults(func=cmd_link)

    # unlink
    p_unlink = subparsers.add_parser("unlink", help="Remove a link between memories")
    p_unlink.add_argument("source", type=int, help="Source memory ID")
    p_unlink.add_argument("target", type=int, help="Target memory ID")
    p_unlink.set_defaults(func=cmd_unlink)

    # links
    p_links = subparsers.add_parser("links", help="Show links for a memory")
    p_links.add_argument("id", type=int, help="Memory ID")
    p_links.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_links.set_defaults(func=cmd_links)

    # types
    p_types = subparsers.add_parser("types", help="List memory types in use")
    p_types.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_types.set_defaults(func=cmd_types)

    # ========== v2.0 Commands ==========

    # topics
    p_topics = subparsers.add_parser("topics", help="List or manage topics")
    p_topics.add_argument("--set", nargs=2, metavar=("ID", "TOPIC"), help="Set topic for memory")
    p_topics.add_argument("--get", metavar="TOPIC", help="Get memories by topic")
    p_topics.add_argument("--limit", "-l", type=int, default=50, help="Limit results")
    p_topics.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_topics.set_defaults(func=cmd_topics)

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge multiple memories into one")
    p_merge.add_argument("ids", type=int, nargs="+", help="Memory IDs to merge")
    p_merge.add_argument("--content", "-c", required=True, help="Content for merged memory")
    p_merge.add_argument("--archive", "-a", action="store_true", help="Archive source memories")
    p_merge.set_defaults(func=cmd_merge)

    # children
    p_children = subparsers.add_parser("children", help="List child memories of a parent")
    p_children.add_argument("id", type=int, help="Parent memory ID")
    p_children.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_children.set_defaults(func=cmd_children)

    # parent
    p_parent = subparsers.add_parser("parent", help="Set or show parent of a memory")
    p_parent.add_argument("id", type=int, help="Memory ID")
    p_parent.add_argument("--set", type=int, metavar="PARENT_ID", help="Set parent (0 to remove)")
    p_parent.set_defaults(func=cmd_parent)

    # confidence
    p_conf = subparsers.add_parser("confidence", help="Set confidence level for a memory")
    p_conf.add_argument("id", type=int, help="Memory ID")
    p_conf.add_argument("value", type=float, help="Confidence value (0.0-1.0)")
    p_conf.set_defaults(func=cmd_confidence)

    # conflicts
    p_conflicts = subparsers.add_parser("conflicts", help="Check or list memory conflicts")
    p_conflicts.add_argument("--check", type=int, metavar="ID", help="Check conflicts for memory")
    p_conflicts.add_argument("--resolve", nargs=2, type=int, metavar=("CONFLICT_ID", "WINNER_ID"),
                             help="Resolve conflict with winning memory")
    p_conflicts.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    p_conflicts.add_argument("--type", choices=["potential", "confirmed", "resolved"], help="Filter by type")
    p_conflicts.add_argument("--note", help="Note when resolving")
    p_conflicts.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_conflicts.set_defaults(func=cmd_conflicts)

    # search-history
    p_shist = subparsers.add_parser("search-history", help="Show search history")
    p_shist.add_argument("--popular", "-p", action="store_true", help="Show popular queries")
    p_shist.add_argument("--session", help="Filter by session ID")
    p_shist.add_argument("--limit", "-l", type=int, default=20)
    p_shist.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_shist.set_defaults(func=cmd_search_history)

    # export-md
    p_expmd = subparsers.add_parser("export-md", help="Export memories as Markdown")
    p_expmd.add_argument("--output", "-o", help="Output file (stdout if not specified)")
    p_expmd.add_argument("--archived", "-a", action="store_true", help="Include archived")
    p_expmd.set_defaults(func=cmd_export_md)

    # export-csv
    p_expcsv = subparsers.add_parser("export-csv", help="Export memories as CSV")
    p_expcsv.add_argument("--output", "-o", help="Output file (stdout if not specified)")
    p_expcsv.add_argument("--archived", "-a", action="store_true", help="Include archived")
    p_expcsv.set_defaults(func=cmd_export_csv)

    # ========== v2.1 Decay Commands ==========

    # decay
    p_decay = subparsers.add_parser("decay", help="Manage memory decay/aging")
    p_decay.add_argument("--status", type=int, metavar="ID", help="Check decay status for memory")
    p_decay.add_argument("--run", action="store_true", help="Run decay process")
    p_decay.add_argument("--dry-run", action="store_true", help="Preview without archiving")
    p_decay.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_decay.set_defaults(func=cmd_decay)

    # protect
    p_protect = subparsers.add_parser("protect", help="Protect a memory from decay")
    p_protect.add_argument("id", type=int, help="Memory ID")
    p_protect.add_argument("--remove", "-r", action="store_true", help="Remove protection")
    p_protect.set_defaults(func=cmd_protect)

    # ========== v2.7 Session Persistence Commands ==========

    # session-save
    p_sess_save = subparsers.add_parser("session-save", help="Save current session state")
    p_sess_save.add_argument("title", help="Short title for the session")
    p_sess_save.add_argument("--summary", "-s", help="Summary of what was being worked on")
    p_sess_save.add_argument("--task", "-t", help="Detailed task description")
    p_sess_save.add_argument("--progress", help="Progress JSON: {completed: N, total: M, items: [...]}")
    p_sess_save.add_argument("--file", "-f", action="append", help="File touched (can repeat)")
    p_sess_save.add_argument("--decision", "-d", action="append", help="Decision made (can repeat)")
    p_sess_save.add_argument("--next", "-n", action="append", help="Next step (can repeat)")
    p_sess_save.add_argument("--blocker", "-b", action="append", help="Blocker (can repeat)")
    p_sess_save.add_argument("--project", "-p", help="Project name")
    p_sess_save.add_argument("--notes", help="Additional notes")
    p_sess_save.add_argument("--tag", action="append", help="Tag (can repeat)")
    p_sess_save.add_argument("--format", choices=["text", "json"], default="text")
    p_sess_save.set_defaults(func=cmd_session_save)

    # session-resume
    p_sess_resume = subparsers.add_parser("session-resume", help="Get resume prompt for a session")
    p_sess_resume.add_argument("--id", type=int, help="Session ID (default: most recent)")
    p_sess_resume.add_argument("--project", "-p", help="Filter by project")
    p_sess_resume.set_defaults(func=cmd_session_resume)

    # session-list
    p_sess_list = subparsers.add_parser("session-list", help="List saved sessions")
    p_sess_list.add_argument("--project", "-p", help="Filter by project")
    p_sess_list.add_argument("--limit", "-l", type=int, default=10, help="Max sessions")
    p_sess_list.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_sess_list.set_defaults(func=cmd_session_list)

    # session-get
    p_sess_get = subparsers.add_parser("session-get", help="Get session details")
    p_sess_get.add_argument("id", type=int, help="Session ID")
    p_sess_get.add_argument("--format", "-f", choices=["text", "json"], default="text")
    p_sess_get.set_defaults(func=cmd_session_get)

    # ========== Hooks Commands ==========

    # hooks-install
    p_hooks_install = subparsers.add_parser("hooks-install", help="Install Claude Code hooks for auto session capture")
    p_hooks_install.set_defaults(func=cmd_hooks_install)

    # hooks-status
    p_hooks_status = subparsers.add_parser("hooks-status", help="Check if Claude Code hooks are installed")
    p_hooks_status.set_defaults(func=cmd_hooks_status)

    # hooks-uninstall
    p_hooks_uninstall = subparsers.add_parser("hooks-uninstall", help="Remove Claude Code hooks")
    p_hooks_uninstall.set_defaults(func=cmd_hooks_uninstall)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
