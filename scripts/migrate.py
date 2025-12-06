#!/usr/bin/env python3
"""
Re-embed memories after model change.
"""

import sys
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_memory import get_config, get_store, get_database, get_embedding
from llama_memory.database import embedding_to_blob


def migrate(dry_run: bool = False, batch_size: int = 10):
    """Re-embed memories that were embedded with a different model."""
    config = get_config()
    store = get_store(config)
    db = get_database(config)

    current_model = config.embedding_model_version

    # Find memories needing re-embedding
    conn = db.conn
    cursor = conn.execute("""
        SELECT id, content, embedding_model
        FROM memories
        WHERE (embedding_model IS NULL OR embedding_model != ?)
          AND archived = 0
    """, (current_model,))

    memories = cursor.fetchall()
    total = len(memories)

    if total == 0:
        print("All memories are up to date.")
        return 0

    print(f"Found {total} memories to re-embed with {current_model}")

    if dry_run:
        print("Dry run - no changes made.")
        return 0

    migrated = 0
    errors = 0

    for memory in memories:
        memory_id = memory['id']
        content = memory['content']
        old_model = memory['embedding_model'] or 'unknown'

        try:
            # Generate new embedding
            embedding = get_embedding(content, config)
            embedding_blob = embedding_to_blob(embedding)

            # Update
            conn.execute(
                "UPDATE memory_embeddings SET embedding = ? WHERE memory_id = ?",
                (embedding_blob, memory_id)
            )
            conn.execute(
                "UPDATE memories SET embedding_model = ? WHERE id = ?",
                (current_model, memory_id)
            )

            migrated += 1
            print(f"  [{migrated}/{total}] Memory {memory_id}: {old_model} -> {current_model}")

            # Commit in batches
            if migrated % batch_size == 0:
                conn.commit()

        except Exception as e:
            errors += 1
            print(f"  [ERROR] Memory {memory_id}: {e}")

    conn.commit()

    print(f"\nMigration complete: {migrated} successful, {errors} errors")
    return 0 if errors == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Re-embed memories after model change")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Commit every N memories")

    args = parser.parse_args()
    return migrate(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
