"""Tests for MemoryBank."""

import pytest
import time
import torch

from cortex_net.memory_bank import MemoryBank, Memory


DIM = 32


@pytest.fixture
def bank(tmp_path):
    return MemoryBank(db_path=tmp_path / "memory.db", blob_dir=tmp_path / "blobs")


class TestBasicOperations:
    def test_add_and_count(self, bank):
        bank.add_text("Test memory", torch.randn(DIM).tolist())
        assert len(bank) == 1

    def test_add_returns_id(self, bank):
        mid = bank.add_text("Test", torch.randn(DIM).tolist())
        assert isinstance(mid, str) and len(mid) == 16

    def test_get_by_id(self, bank):
        mid = bank.add_text("Hello world", torch.randn(DIM).tolist(), source="user", importance=0.8)
        mem = bank.get(mid)
        assert mem is not None
        assert mem.text == "Hello world"
        assert mem.source == "user"
        assert mem.importance == 0.8
        assert mem.content_type == "text"

    def test_get_texts(self, bank):
        id1 = bank.add_text("First", torch.randn(DIM).tolist())
        id2 = bank.add_text("Second", torch.randn(DIM).tolist())
        texts = bank.get_texts([id2, id1])  # reversed order
        assert texts == ["Second", "First"]

    def test_get_all_embeddings(self, bank):
        bank.add_text("A", torch.randn(DIM).tolist())
        bank.add_text("B", torch.randn(DIM).tolist())
        ids, embs = bank.get_all_embeddings()
        assert len(ids) == 2
        assert embs.shape == (2, DIM)

    def test_empty_bank(self, bank):
        ids, embs = bank.get_all_embeddings()
        assert ids == []
        assert embs is None

    def test_accepts_tensor(self, bank):
        mid = bank.add_text("Tensor input", torch.randn(DIM))
        mem = bank.get(mid)
        assert len(mem.embedding) == DIM


class TestContentTypes:
    def test_text_content(self, bank):
        mid = bank.add_text("Just text", torch.randn(DIM).tolist())
        mem = bank.get(mid)
        assert mem.content_type == "text"
        assert mem.content_ref is None

    def test_blob_content(self, bank):
        mid = bank.add_blob(
            description="Architecture diagram",
            embedding=torch.randn(DIM).tolist(),
            blob_data=b"fake image data",
            filename="arch.png",
            content_type="image",
            source="user",
        )
        mem = bank.get(mid)
        assert mem.content_type == "image"
        assert mem.content_ref is not None
        assert "arch" in mem.content_ref

        # Blob should exist on disk
        blob_path = bank.blob_dir / mem.content_ref
        assert blob_path.exists()
        assert blob_path.read_bytes() == b"fake image data"

    def test_structured_content(self, bank):
        mid = bank.add(
            "API response data",
            torch.randn(DIM).tolist(),
            content_type="structured",
            metadata={"schema": "api_response", "status": 200},
        )
        mem = bank.get(mid)
        assert mem.content_type == "structured"
        assert mem.metadata["status"] == 200


class TestRetrieval:
    def test_cosine_retrieval(self, bank):
        # Add memories with known embeddings
        target = torch.randn(DIM)
        bank.add_text("Relevant", target.tolist())
        bank.add_text("Irrelevant", (-target).tolist())  # opposite direction
        bank.add_text("Noise", torch.randn(DIM).tolist())

        results = bank.retrieve(target, k=2)
        assert len(results) == 2
        assert results[0].memory.text == "Relevant"
        assert results[0].score > results[1].score

    def test_retrieval_updates_access_count(self, bank):
        mid = bank.add_text("Will be accessed", torch.randn(DIM).tolist())
        bank.retrieve(torch.randn(DIM), k=1)
        mem = bank.get(mid)
        assert mem.access_count >= 1

    def test_filter_by_content_type(self, bank):
        bank.add_text("Text memory", torch.randn(DIM).tolist())
        bank.add("Image desc", torch.randn(DIM).tolist(), content_type="image")

        results = bank.retrieve(torch.randn(DIM), k=10, content_types=["text"])
        assert all(r.memory.content_type == "text" for r in results)

    def test_filter_by_importance(self, bank):
        bank.add_text("Important", torch.randn(DIM).tolist(), importance=0.9)
        bank.add_text("Unimportant", torch.randn(DIM).tolist(), importance=0.1)

        results = bank.retrieve(torch.randn(DIM), k=10, min_importance=0.5)
        assert all(r.memory.importance >= 0.5 for r in results)


class TestMaintenance:
    def test_consolidate_similar(self, bank):
        emb = torch.randn(DIM).tolist()
        # Two nearly identical memories
        bank.add_text("Deploy with Kubernetes", emb)
        bank.add_text("Deploy using Kubernetes", emb)  # same embedding = similarity 1.0
        bank.add_text("Something different", torch.randn(DIM).tolist())

        merged = bank.consolidate(similarity_threshold=0.99)
        assert merged >= 1
        assert bank.count() == 2  # one was deactivated

    def test_prune(self, bank):
        for i in range(10):
            bank.add_text(f"Memory {i}", torch.randn(DIM).tolist(), importance=i * 0.1)

        pruned = bank.prune(keep=5)
        assert pruned == 5
        assert bank.count() == 5

        # Should have kept the most important ones
        ids, _ = bank.get_all_embeddings()
        for mid in ids:
            mem = bank.get(mid)
            assert mem.importance >= 0.4  # kept top 5 (0.5-0.9)

    def test_stats(self, bank):
        bank.add_text("A", torch.randn(DIM).tolist(), source="user")
        bank.add_text("B", torch.randn(DIM).tolist(), source="assistant")
        bank.add("C", torch.randn(DIM).tolist(), content_type="image", source="user")

        stats = bank.stats()
        assert stats["active_memories"] == 3
        assert stats["content_types"]["text"] == 2
        assert stats["content_types"]["image"] == 1
        assert stats["sources"]["user"] == 2


class TestPersistence:
    def test_survives_reopen(self, tmp_path):
        db_path = tmp_path / "memory.db"

        bank1 = MemoryBank(db_path=db_path)
        bank1.add_text("Persistent memory", torch.randn(DIM).tolist(), importance=0.9)
        bank1.close()

        bank2 = MemoryBank(db_path=db_path)
        assert len(bank2) == 1
        ids, embs = bank2.get_all_embeddings()
        assert embs.shape == (1, DIM)
        mem = bank2.get(ids[0])
        assert mem.text == "Persistent memory"
        assert mem.importance == 0.9
        bank2.close()


class TestParentChild:
    def test_derived_memory(self, bank):
        parent_id = bank.add_text("Full meeting notes about deployment...", torch.randn(DIM).tolist())
        child_id = bank.add_text(
            "Summary: decided to use blue-green for all critical services",
            torch.randn(DIM).tolist(),
            parent_id=parent_id,
            source="extracted",
        )
        child = bank.get(child_id)
        assert child.parent_id == parent_id
        assert child.source == "extracted"
