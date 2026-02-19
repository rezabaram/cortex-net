"""Embedding Store — persistent cache for text embeddings.

Maps text → embedding vector, persisted to disk as a JSONL file.
Used by the training loop so the Memory Gate trains on real
sentence-transformer embeddings, not synthetic vectors.

Design:
- In-memory dict backed by append-only JSONL on disk
- Deduplicates by text content (hash key)
- Bulk encode via sentence-transformers
- Survives restarts (load from disk on init)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer


def _text_key(text: str) -> str:
    """Stable hash key for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingStore:
    """Persistent embedding cache.

    Args:
        store_path: Path to the JSONL backing file.
        encoder: SentenceTransformer model (shared with pipeline).
        device: PyTorch device.
    """

    def __init__(
        self,
        store_path: str | Path,
        encoder: SentenceTransformer | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SentenceTransformer(model_name, device=device)

        # In-memory cache: key -> (text, embedding_list)
        self._cache: dict[str, tuple[str, list[float]]] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load existing embeddings from JSONL file."""
        if not self.store_path.exists():
            return
        with open(self.store_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._cache[entry["key"]] = (entry["text"], entry["embedding"])
                except (json.JSONDecodeError, KeyError):
                    continue

    def _append_to_disk(self, key: str, text: str, embedding: list[float]) -> None:
        """Append a single entry to the JSONL file."""
        entry = {"key": key, "text": text, "embedding": embedding}
        with open(self.store_path, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def get(self, text: str) -> torch.Tensor | None:
        """Get cached embedding for text, or None if not cached."""
        key = _text_key(text)
        if key in self._cache:
            return torch.tensor(self._cache[key][1], device=self.device)
        return None

    def encode(self, text: str) -> torch.Tensor:
        """Get or compute embedding for a single text."""
        key = _text_key(text)
        if key in self._cache:
            return torch.tensor(self._cache[key][1], device=self.device)

        embedding = self.encoder.encode(text, convert_to_tensor=True).cpu().tolist()
        self._cache[key] = (text, embedding)
        self._append_to_disk(key, text, embedding)
        return torch.tensor(embedding, device=self.device)

    def encode_batch(self, texts: Sequence[str]) -> torch.Tensor:
        """Get or compute embeddings for multiple texts.

        Only encodes texts not already cached.
        """
        results: list[torch.Tensor] = []
        to_encode: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            key = _text_key(text)
            if key in self._cache:
                results.append(torch.tensor(self._cache[key][1], device=self.device))
            else:
                results.append(None)  # type: ignore
                to_encode.append((i, text))

        if to_encode:
            indices, new_texts = zip(*to_encode)
            new_embeddings = self.encoder.encode(
                list(new_texts), convert_to_tensor=True
            )
            for idx, text, emb in zip(indices, new_texts, new_embeddings):
                emb_list = emb.cpu().tolist()
                key = _text_key(text)
                self._cache[key] = (text, emb_list)
                self._append_to_disk(key, text, emb_list)
                results[idx] = torch.tensor(emb_list, device=self.device)

        return torch.stack(results).to(self.device)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, text: str) -> bool:
        return _text_key(text) in self._cache
