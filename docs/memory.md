# Memory System

cortex-net uses a SQLite-backed memory system (`MemoryBank`) designed to handle text today and extend to multimodal content (images, files, audio) without architectural changes.

## Core Design

Every memory is a text description + embedding + metadata. Retrieval is always via embeddings (through the Memory Gate or cosine fallback). The actual content — whether text, an image, or a file — is stored separately.

```python
@dataclass
class Memory:
    id: str                          # UUID
    text: str                        # Human-readable description
    embedding: list[float]           # 384-dim vector for retrieval
    content_type: str                # text | image | file | audio | video | structured
    content_ref: str | None          # Path to blob (None for pure text)
    source: str                      # user | assistant | system | seeded | extracted
    importance: float                # [0, 1] — learned from access patterns
    access_count: int                # How often this memory was retrieved
    created_at: float
    last_accessed: float
    tags: list[str]
    metadata: dict
    parent_id: str | None            # For derived memories (summaries)
    active: bool                     # Soft delete / decay
```

## Storage Layout

```
state/
├── memory.db          # SQLite: records + binary embeddings + metadata
└── blobs/             # Binary content, organized by date
    └── 2026-02-19/
        ├── arch-diagram.png
        └── meeting-notes.pdf
```

Embeddings are stored as packed binary (4 bytes per float, ~1.5KB for 384-dim). No re-encoding on load.

SQLite runs in WAL mode for concurrent read performance.

## Usage

### Text Memories

```python
from cortex_net.memory_bank import MemoryBank

bank = MemoryBank(db_path="./state/memory.db")

# Add with pre-computed embedding
bank.add_text(
    "We use Kubernetes for all production deployments",
    embedding=encoder.encode("We use Kubernetes..."),
    source="seeded",
    importance=0.7,
    tags=["infrastructure", "deployment"],
)
```

### Binary Content (Images, Files)

```python
# Store an image — the description gets embedded, the image goes to blobs/
bank.add_blob(
    description="Architecture diagram showing 3 microservices and their connections",
    embedding=encoder.encode("Architecture diagram..."),
    blob_data=open("arch.png", "rb").read(),
    filename="arch.png",
    content_type="image",
    source="user",
)
```

### Structured Data

```python
bank.add(
    "API health check response showing all services healthy",
    embedding=encoder.encode("API health check..."),
    content_type="structured",
    metadata={"schema": "health_check", "services": 5, "all_healthy": True},
)
```

### Derived Memories

Create summaries or extractions linked to their source:

```python
parent_id = bank.add_text("Full 2-hour meeting transcript about deployment strategy...")

bank.add_text(
    "Decision: use blue-green deployments for all critical services",
    embedding=...,
    parent_id=parent_id,       # Links to the full transcript
    source="extracted",
    importance=0.8,            # Summary is more important than raw transcript
)
```

## Retrieval

Retrieval uses the Memory Gate (learned relevance) or cosine similarity (fallback):

```python
results = bank.retrieve(
    query_embedding=situation_embedding,
    k=5,
    gate=memory_gate,              # Learned retrieval (or None for cosine)
    content_types=["text", "image"],  # Filter by type
    min_importance=0.3,             # Skip low-importance memories
)

for r in results:
    print(f"[{r.rank}] score={r.score:.3f} | {r.memory.text}")
    if r.memory.content_ref:
        print(f"    → blob: {r.memory.content_ref}")
```

Access counts are automatically updated on retrieval — frequently retrieved memories gain importance over time.

## Maintenance

### Importance Decay

Unused memories gradually lose importance:

```python
bank.decay_importance()
# Memories not accessed recently get importance *= exp(-rate * days)
# Floor of 0.01 — nothing is completely forgotten
```

### Consolidation

Merge near-duplicate memories:

```python
merged = bank.consolidate(similarity_threshold=0.95)
# "Deploy with Kubernetes" + "Deploy using Kubernetes" → merged into one
# Keeps the higher-importance version, combines access counts
```

### Pruning

Remove lowest-importance memories beyond capacity:

```python
pruned = bank.prune(keep=5000)
# Removes the least important, least accessed memories
# Soft delete (active=False) — recoverable
```

### Stats

```python
bank.stats()
# {
#   'active_memories': 32,
#   'total_memories': 35,
#   'archived': 3,
#   'avg_importance': 0.542,
#   'avg_access_count': 2.1,
#   'content_types': {'text': 28, 'image': 3, 'structured': 1},
#   'sources': {'seeded': 10, 'user': 12, 'assistant': 8, 'extracted': 2},
# }
```

## How It Learns

1. **Seeded memories** start at importance 0.7
2. **Auto-memorized** exchanges start at 0.4-0.5
3. **Access count** increases each time a memory is retrieved
4. **Importance decays** for memories that aren't accessed
5. **Consolidation** merges near-duplicates
6. **Pruning** removes the lowest-value memories when capacity is reached
7. The **Memory Gate** (trained via online learning) gets better at surfacing the right memories over time

The result: the agent naturally accumulates useful knowledge and forgets noise, without manual curation.
