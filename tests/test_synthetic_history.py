"""Tests for synthetic history generator."""

import json
import tempfile
from pathlib import Path

from cortex_net.synthetic_history import HistoryGenerator, SyntheticHistory, TOPICS, TOPIC_MAP


def test_generate_basic():
    """Generator produces turns and queries."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=1, turns_per_day=(4, 8))
    assert len(history.turns) > 100
    assert len(history.queries) > 5
    assert history.metadata["months"] == 1


def test_turn_structure():
    """Each turn has required fields."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=1, turns_per_day=(4, 8))
    for turn in history.turns[:20]:
        assert turn.id.startswith("t")
        assert turn.role in ("user", "assistant")
        assert len(turn.content) > 10
        assert turn.timestamp > 0
        assert turn.topic_id in TOPIC_MAP
        assert turn.session_id.startswith("s")


def test_alternating_roles():
    """Turns alternate user/assistant within sessions."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=1, turns_per_day=(4, 8))

    # Group by session
    sessions: dict[str, list] = {}
    for turn in history.turns:
        sessions.setdefault(turn.session_id, []).append(turn)

    for sid, turns in list(sessions.items())[:10]:
        for i, turn in enumerate(turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert turn.role == expected_role, f"Session {sid}, turn {i}: expected {expected_role}, got {turn.role}"


def test_chronological_order():
    """Turns are in chronological order."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=1, turns_per_day=(4, 8))
    for i in range(1, len(history.turns)):
        assert history.turns[i].timestamp >= history.turns[i - 1].timestamp


def test_topic_diversity():
    """History covers multiple topics."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=2, turns_per_day=(5, 15))
    topics_seen = {t.topic_id for t in history.turns}
    assert len(topics_seen) >= 8, f"Only {len(topics_seen)} topics: {topics_seen}"


def test_category_diversity():
    """History covers multiple categories."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=2, turns_per_day=(5, 15))
    categories = {t.category for t in history.turns}
    assert "coding" in categories
    assert "debugging" in categories
    assert "architecture" in categories


def test_query_types():
    """Ground truth queries cover different recall types."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=3, turns_per_day=(8, 20))
    categories = {q.category for q in history.queries}
    assert "topic_recall" in categories
    assert len(history.queries) >= 10


def test_query_ground_truth_valid():
    """Query ground truth turn IDs exist in the history."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=2, turns_per_day=(5, 15))
    all_ids = {t.id for t in history.turns}
    for query in history.queries:
        for tid in query.relevant_turn_ids:
            assert tid in all_ids, f"Query references non-existent turn {tid}"


def test_deterministic():
    """Same seed produces same output."""
    h1 = HistoryGenerator(seed=123).generate(months=1, turns_per_day=(4, 8))
    h2 = HistoryGenerator(seed=123).generate(months=1, turns_per_day=(4, 8))
    assert len(h1.turns) == len(h2.turns)
    assert h1.turns[0].content == h2.turns[0].content
    assert h1.turns[-1].content == h2.turns[-1].content


def test_save_load(tmp_path):
    """History can be saved and loaded."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=1, turns_per_day=(4, 8))
    path = tmp_path / "history.json"
    history.save(path)

    loaded = SyntheticHistory.load(path)
    assert len(loaded.turns) == len(history.turns)
    assert len(loaded.queries) == len(history.queries)
    assert loaded.turns[0].content == history.turns[0].content


def test_scale_6_months():
    """6 months generates 10K+ turns."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=6, turns_per_day=(8, 25))
    assert len(history.turns) >= 2000, f"Only {len(history.turns)} turns for 6 months"
    assert len(history.queries) >= 15
    print(f"\n6-month history: {len(history.turns)} turns, {history.metadata['total_sessions']} sessions, {len(history.queries)} queries")


def test_no_empty_content():
    """No turn has empty or placeholder content."""
    gen = HistoryGenerator(seed=42)
    history = gen.generate(months=1, turns_per_day=(4, 8))
    for turn in history.turns:
        assert "{" not in turn.content, f"Unfilled template in: {turn.content[:80]}"
        assert len(turn.content.strip()) > 5
