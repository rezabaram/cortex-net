"""Tests for EmbeddingStore."""

import pytest
import torch

from cortex_net.embedding_store import EmbeddingStore


@pytest.fixture
def store(tmp_path):
    return EmbeddingStore(tmp_path / "embeddings.jsonl")


def test_encode_returns_tensor(store):
    emb = store.encode("hello world")
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (384,)


def test_cache_hit(store):
    emb1 = store.encode("test text")
    emb2 = store.encode("test text")
    assert torch.allclose(emb1, emb2)
    assert len(store) == 1


def test_encode_batch(store):
    texts = ["first", "second", "third"]
    embs = store.encode_batch(texts)
    assert embs.shape == (3, 384)


def test_persistence(tmp_path):
    path = tmp_path / "embeddings.jsonl"
    store1 = EmbeddingStore(path)
    store1.encode("persistent text")
    assert len(store1) == 1

    store2 = EmbeddingStore(path, encoder=store1.encoder)
    assert len(store2) == 1
    assert "persistent text" in store2


def test_batch_deduplication(store):
    store.encode("already cached")
    embs = store.encode_batch(["already cached", "new text"])
    assert embs.shape == (2, 384)
    assert len(store) == 2
