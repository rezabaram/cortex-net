"""MemoryBank — persistent, extensible memory storage with learned retrieval.

SQLite-backed memory system that stores text, embeddings, metadata, and
content references. Designed to extend to multimodal (images, files, audio)
without changing the retrieval architecture.

Core idea: everything gets a text description + embedding. Retrieval is always
via the Memory Gate on embeddings. The actual content (text, file, image) is
stored separately and referenced by the memory entry.

Storage layout:
    state/
    ├── memory.db          # SQLite: memory records + embeddings + metadata
    └── blobs/             # Binary content (images, files, etc.)
        ├── 2026-02-19/
        │   ├── arch-diagram.png
        │   └── meeting-notes.pdf
        └── ...
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


# --- Data types ---

@dataclass
class Memory:
    """A single memory entry."""

    id: str                          # UUID
    text: str                        # Human-readable description
    embedding: list[float]           # Dense vector for retrieval
    content_type: str = "text"       # text | image | file | audio | video | structured
    content_ref: str | None = None   # Path to blob (None for pure text)
    source: str = "system"           # user | assistant | system | seeded | extracted
    importance: float = 0.5          # [0, 1] — learned from access patterns
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None     # For derived memories (summaries, extractions)
    active: bool = True              # Soft delete / decay


@dataclass
class RetrievalResult:
    """Result of a memory retrieval."""

    memory: Memory
    score: float                     # Relevance score from Memory Gate
    rank: int


# --- Embedding serialization ---

def _pack_embedding(emb: list[float]) -> bytes:
    """Pack a float list into compact binary (4 bytes per float)."""
    return struct.pack(f'{len(emb)}f', *emb)


def _unpack_embedding(data: bytes) -> list[float]:
    """Unpack binary into float list."""
    n = len(data) // 4
    return list(struct.unpack(f'{n}f', data))


# --- MemoryBank ---

class MemoryBank:
    """SQLite-backed memory storage with learned retrieval.

    Features:
    - Persistent embeddings (no re-encoding on load)
    - Metadata per memory (source, importance, access patterns)
    - Importance decay for unused memories
    - Consolidation of similar memories
    - Soft deletion and pruning
    - Content references for multimodal (blobs stored on disk)
    - Full-text and embedding-based retrieval

    Args:
        db_path: Path to SQLite database.
        blob_dir: Directory for binary content (images, files, etc.).
        max_memories: Soft capacity limit (pruning removes lowest-importance beyond this).
        decay_rate: Importance decay per day for unaccessed memories.
    """

    def __init__(
        self,
        db_path: str | Path = "./state/memory.db",
        blob_dir: str | Path | None = None,
        max_memories: int = 10000,
        decay_rate: float = 0.01,
    ) -> None:
        self.db_path = Path(db_path)
        self.blob_dir = Path(blob_dir) if blob_dir else self.db_path.parent / "blobs"
        self.max_memories = max_memories
        self.decay_rate = decay_rate

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'text',
                content_ref TEXT,
                source TEXT NOT NULL DEFAULT 'system',
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                parent_id TEXT,
                active INTEGER NOT NULL DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_content_type ON memories(content_type);
            CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
        """)
        self._conn.commit()

    # --- Core operations ---

    def add(
        self,
        text: str,
        embedding: list[float] | torch.Tensor,
        content_type: str = "text",
        content_ref: str | None = None,
        source: str = "system",
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        parent_id: str | None = None,
    ) -> str:
        """Add a memory. Returns the memory ID."""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().tolist()

        mem_id = uuid.uuid4().hex[:16]
        now = time.time()

        self._conn.execute(
            """INSERT INTO memories
               (id, text, embedding, content_type, content_ref, source,
                importance, access_count, created_at, last_accessed,
                tags, metadata, parent_id, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, 1)""",
            (
                mem_id,
                text,
                _pack_embedding(embedding),
                content_type,
                content_ref,
                source,
                importance,
                now,
                now,
                json.dumps(tags or []),
                json.dumps(metadata or {}),
                parent_id,
            ),
        )
        self._conn.commit()
        return mem_id

    def add_text(self, text: str, embedding: list[float] | torch.Tensor, **kwargs) -> str:
        """Convenience: add a text memory."""
        return self.add(text, embedding, content_type="text", **kwargs)

    def add_blob(
        self,
        description: str,
        embedding: list[float] | torch.Tensor,
        blob_data: bytes,
        filename: str,
        content_type: str = "file",
        **kwargs,
    ) -> str:
        """Add a binary content memory (image, file, etc.).

        Stores the blob on disk and creates a memory entry referencing it.
        """
        # Store blob
        from datetime import date
        day_dir = self.blob_dir / date.today().isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        blob_path = day_dir / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        blob_path.write_bytes(blob_data)

        return self.add(
            description, embedding,
            content_type=content_type,
            content_ref=str(blob_path.relative_to(self.blob_dir)),
            **kwargs,
        )

    def get(self, mem_id: str) -> Memory | None:
        """Get a memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_all_embeddings(self) -> tuple[list[str], torch.Tensor] | tuple[list[str], None]:
        """Get all active memory IDs and embeddings as a tensor.

        Returns:
            (ids, embeddings_tensor) or ([], None) if empty.
        """
        rows = self._conn.execute(
            "SELECT id, embedding FROM memories WHERE active = 1"
        ).fetchall()

        if not rows:
            return [], None

        ids = [r[0] for r in rows]
        embs = torch.tensor([_unpack_embedding(r[1]) for r in rows])
        return ids, embs

    def get_texts(self, mem_ids: list[str]) -> list[str]:
        """Get memory texts by IDs, preserving order."""
        if not mem_ids:
            return []
        placeholders = ",".join("?" * len(mem_ids))
        rows = self._conn.execute(
            f"SELECT id, text FROM memories WHERE id IN ({placeholders})",
            mem_ids,
        ).fetchall()
        id_to_text = {r[0]: r[1] for r in rows}
        return [id_to_text.get(mid, "") for mid in mem_ids]

    def get_memories(self, mem_ids: list[str]) -> list[Memory]:
        """Get full Memory objects by IDs."""
        if not mem_ids:
            return []
        placeholders = ",".join("?" * len(mem_ids))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            mem_ids,
        ).fetchall()
        id_to_mem = {self._row_to_memory(r).id: self._row_to_memory(r) for r in rows}
        return [id_to_mem[mid] for mid in mem_ids if mid in id_to_mem]

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
        gate=None,
        content_types: list[str] | None = None,
        min_importance: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve top-k relevant memories.

        Uses Memory Gate if provided, otherwise cosine similarity.

        Args:
            query_embedding: Query embedding (situation or text).
            k: Number of results.
            gate: Optional MemoryGate for learned retrieval.
            content_types: Filter by content type (None = all).
            min_importance: Minimum importance threshold.

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        ids, embs = self.get_all_embeddings()
        if embs is None:
            return []

        # Filter by content type and importance
        if content_types or min_importance > 0:
            mask_ids = set()
            query_parts = ["SELECT id FROM memories WHERE active = 1"]
            params: list = []

            if content_types:
                placeholders = ",".join("?" * len(content_types))
                query_parts.append(f"AND content_type IN ({placeholders})")
                params.extend(content_types)

            if min_importance > 0:
                query_parts.append("AND importance >= ?")
                params.append(min_importance)

            rows = self._conn.execute(" ".join(query_parts), params).fetchall()
            mask_ids = {r[0] for r in rows}

            # Filter ids and embeddings
            mask = [i for i, mid in enumerate(ids) if mid in mask_ids]
            if not mask:
                return []
            ids = [ids[i] for i in mask]
            embs = embs[mask]

        k = min(k, len(ids))

        # Score
        if gate is not None:
            with torch.no_grad():
                idx, scores = gate.select_top_k(query_embedding, embs, k=k)
            selected_ids = [ids[i] for i in idx.tolist()]
            selected_scores = scores.tolist()
        else:
            # Cosine similarity fallback
            import torch.nn.functional as F
            scores = F.cosine_similarity(query_embedding.unsqueeze(0), embs, dim=1)
            top_k = scores.argsort(descending=True)[:k]
            selected_ids = [ids[i] for i in top_k.tolist()]
            selected_scores = [scores[i].item() for i in top_k.tolist()]

        # Update access counts
        now = time.time()
        for mid in selected_ids:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, mid),
            )
        self._conn.commit()

        # Build results
        memories = self.get_memories(selected_ids)
        results = []
        for rank, (mem, score) in enumerate(zip(memories, selected_scores)):
            results.append(RetrievalResult(memory=mem, score=score, rank=rank))
        return results

    # --- Maintenance ---

    def decay_importance(self) -> int:
        """Decay importance of unaccessed memories. Returns count affected."""
        now = time.time()
        day_seconds = 86400.0

        rows = self._conn.execute(
            "SELECT id, importance, last_accessed FROM memories WHERE active = 1"
        ).fetchall()

        updated = 0
        for mem_id, importance, last_accessed in rows:
            days_since = (now - last_accessed) / day_seconds
            if days_since < 1:
                continue
            new_importance = importance * math.exp(-self.decay_rate * days_since)
            new_importance = max(0.01, new_importance)  # Floor
            if abs(new_importance - importance) > 0.001:
                self._conn.execute(
                    "UPDATE memories SET importance = ? WHERE id = ?",
                    (new_importance, mem_id),
                )
                updated += 1

        self._conn.commit()
        return updated

    def consolidate(self, similarity_threshold: float = 0.95) -> int:
        """Merge very similar memories. Returns count merged."""
        ids, embs = self.get_all_embeddings()
        if embs is None or len(ids) < 2:
            return 0

        import torch.nn.functional as F
        # Pairwise cosine similarity (only upper triangle)
        sims = F.cosine_similarity(embs.unsqueeze(0), embs.unsqueeze(1), dim=2)

        merged = 0
        merged_ids = set()

        for i in range(len(ids)):
            if ids[i] in merged_ids:
                continue
            for j in range(i + 1, len(ids)):
                if ids[j] in merged_ids:
                    continue
                if sims[i, j] > similarity_threshold:
                    # Keep the one with higher importance/access_count
                    mem_i = self.get(ids[i])
                    mem_j = self.get(ids[j])
                    if mem_i and mem_j:
                        keep, remove = (mem_i, mem_j) if mem_i.importance >= mem_j.importance else (mem_j, mem_i)
                        # Merge: combine text if different, sum access counts
                        if keep.text != remove.text:
                            new_text = f"{keep.text} (also: {remove.text})"
                            self._conn.execute(
                                "UPDATE memories SET text = ? WHERE id = ?",
                                (new_text, keep.id),
                            )
                        self._conn.execute(
                            "UPDATE memories SET access_count = access_count + ? WHERE id = ?",
                            (remove.access_count, keep.id),
                        )
                        self._conn.execute(
                            "UPDATE memories SET active = 0 WHERE id = ?",
                            (remove.id,),
                        )
                        merged_ids.add(remove.id)
                        merged += 1

        self._conn.commit()
        return merged

    def prune(self, keep: int | None = None) -> int:
        """Remove lowest-importance memories beyond capacity. Returns count pruned."""
        keep = keep or self.max_memories
        count = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE active = 1"
        ).fetchone()[0]

        if count <= keep:
            return 0

        to_remove = count - keep
        rows = self._conn.execute(
            "SELECT id FROM memories WHERE active = 1 ORDER BY importance ASC, last_accessed ASC LIMIT ?",
            (to_remove,),
        ).fetchall()

        for (mem_id,) in rows:
            self._conn.execute(
                "UPDATE memories SET active = 0 WHERE id = ?", (mem_id,)
            )

        self._conn.commit()
        return len(rows)

    # --- Stats ---

    def count(self, active_only: bool = True) -> int:
        """Count memories."""
        where = "WHERE active = 1" if active_only else ""
        return self._conn.execute(f"SELECT COUNT(*) FROM memories {where}").fetchone()[0]

    def stats(self) -> dict[str, Any]:
        """Get memory bank statistics."""
        active = self.count(active_only=True)
        total = self.count(active_only=False)

        row = self._conn.execute(
            "SELECT AVG(importance), AVG(access_count), MIN(created_at), MAX(created_at) "
            "FROM memories WHERE active = 1"
        ).fetchone()

        type_counts = dict(self._conn.execute(
            "SELECT content_type, COUNT(*) FROM memories WHERE active = 1 GROUP BY content_type"
        ).fetchall())

        source_counts = dict(self._conn.execute(
            "SELECT source, COUNT(*) FROM memories WHERE active = 1 GROUP BY source"
        ).fetchall())

        return {
            "active_memories": active,
            "total_memories": total,
            "archived": total - active,
            "avg_importance": round(row[0] or 0, 3),
            "avg_access_count": round(row[1] or 0, 1),
            "content_types": type_counts,
            "sources": source_counts,
        }

    def __len__(self) -> int:
        return self.count()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # --- Internal ---

    def _row_to_memory(self, row: tuple) -> Memory:
        return Memory(
            id=row[0],
            text=row[1],
            embedding=_unpack_embedding(row[2]),
            content_type=row[3],
            content_ref=row[4],
            source=row[5],
            importance=row[6],
            access_count=row[7],
            created_at=row[8],
            last_accessed=row[9],
            tags=json.loads(row[10]),
            metadata=json.loads(row[11]),
            parent_id=row[12],
            active=bool(row[13]),
        )
