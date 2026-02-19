"""Tests for scale benchmark."""

import tempfile
from pathlib import Path

from cortex_net.scale_benchmark import (
    ScaleBenchmark,
    precision_at_k,
    recall_at_k,
    mrr,
    estimate_tokens,
    CosineTopK,
    BM25Retrieval,
    RecencyRetrieval,
    RandomRetrieval,
)
import torch


# ── Metric tests ──────────────────────────────────────────────────────────────

def test_precision_at_k():
    relevant = {"a", "b", "c"}
    assert precision_at_k(["a", "x", "b"], relevant, 3) == 2 / 3
    assert precision_at_k(["x", "y", "z"], relevant, 3) == 0.0
    assert precision_at_k(["a", "b", "c"], relevant, 3) == 1.0
    assert precision_at_k([], relevant, 3) == 0.0


def test_recall_at_k():
    relevant = {"a", "b", "c"}
    assert recall_at_k(["a", "b"], relevant, 2) == 2 / 3
    assert recall_at_k(["a", "b", "c"], relevant, 3) == 1.0
    assert recall_at_k(["x", "y"], relevant, 2) == 0.0


def test_mrr():
    relevant = {"b", "c"}
    assert mrr(["a", "b", "c"], relevant) == 0.5  # b at position 2
    assert mrr(["b", "a", "c"], relevant) == 1.0  # b at position 1
    assert mrr(["x", "y", "z"], relevant) == 0.0


def test_estimate_tokens():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("a" * 400) == 100


# ── Retrieval method tests ────────────────────────────────────────────────────

def test_cosine_topk():
    ids = ["t1", "t2", "t3"]
    embs = torch.randn(3, 8)
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    method = CosineTopK(ids, embs)
    query = torch.nn.functional.normalize(embs[0:1], p=2, dim=1).squeeze(0)
    results = method.retrieve(query, k=2)
    assert len(results) == 2
    assert results[0][0] == "t1"  # most similar to itself


def test_bm25():
    ids = ["t1", "t2", "t3"]
    texts = ["python debugging error", "javascript css styling", "python memory leak"]
    method = BM25Retrieval(ids, texts)
    results = method.retrieve(torch.zeros(8), k=2, query_text="python error debugging")
    assert len(results) == 2
    # t1 should rank highest (most keyword overlap)
    assert results[0][0] == "t1"


def test_recency():
    ids = ["t1", "t2", "t3", "t4", "t5"]
    method = RecencyRetrieval(ids)
    results = method.retrieve(torch.zeros(8), k=3)
    assert len(results) == 3
    assert results[0][0] == "t5"  # most recent first


def test_random_deterministic():
    ids = [f"t{i}" for i in range(100)]
    m1 = RandomRetrieval(ids, seed=42)
    m2 = RandomRetrieval(ids, seed=42)
    r1 = m1.retrieve(torch.zeros(8), k=5)
    r2 = m2.retrieve(torch.zeros(8), k=5)
    assert [r[0] for r in r1] == [r[0] for r in r2]


# ── Integration test ──────────────────────────────────────────────────────────

def test_benchmark_runs():
    """Full benchmark runs end-to-end with small history."""
    bench = ScaleBenchmark(k=3, seed=42)
    bench.load_history(months=1, turns_per_day=(4, 8))
    results = bench.run()

    assert len(results.query_results) > 0
    assert "cortex_net" in results.summaries
    assert "cosine_topk" in results.summaries
    assert "bm25" in results.summaries
    assert "recency" in results.summaries
    assert "random" in results.summaries

    # cortex-net should have results
    cn = results.summaries["cortex_net"]
    assert cn.total_queries > 0
    assert cn.avg_context_tokens > 0
    assert cn.avg_retrieval_ms >= 0

    # Random should generally score lower than cosine
    rnd = results.summaries["random"]
    cos = results.summaries["cosine_topk"]
    # (Not always true for small histories, but at least both should run)
    assert rnd.total_queries == cos.total_queries


def test_benchmark_save_load(tmp_path):
    """Results can be saved and loaded."""
    bench = ScaleBenchmark(k=3, seed=42)
    bench.load_history(months=1, turns_per_day=(4, 8))
    results = bench.run()

    path = tmp_path / "results.json"
    results.save(path)
    assert path.exists()

    import json
    with open(path) as f:
        data = json.load(f)
    assert "summaries" in data
    assert "cortex_net" in data["summaries"]


def test_per_category_breakdown():
    """Results include per-category metrics."""
    bench = ScaleBenchmark(k=3, seed=42)
    bench.load_history(months=2, turns_per_day=(5, 15))
    results = bench.run()

    cn = results.summaries["cortex_net"]
    assert len(cn.by_category) > 0
    assert "topic_recall" in cn.by_category
