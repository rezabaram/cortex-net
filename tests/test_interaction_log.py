"""Tests for InteractionLogger."""

import pytest

from cortex_net.interaction_log import (
    InteractionLogger,
    InteractionRecord,
    SituationData,
    RetrievalData,
    UsageData,
    OutcomeData,
    OutcomeSignal,
)


@pytest.fixture
def logger(tmp_path):
    return InteractionLogger(tmp_path / "logs")


def test_begin_creates_record(logger):
    record = logger.begin(SituationData(message="hello"))
    assert record.situation.message == "hello"
    assert record.id  # has a uuid
    assert record.timestamp  # has a timestamp


def test_commit_and_read(logger):
    record = logger.begin(SituationData(message="test"))
    record.outcome = OutcomeData(signal=OutcomeSignal.POSITIVE, source="user")
    logger.commit(record)

    records = logger.read_all()
    assert len(records) == 1
    assert records[0].situation.message == "test"
    assert records[0].outcome.signal == OutcomeSignal.POSITIVE


def test_append_multiple(logger):
    for i in range(5):
        r = logger.begin(SituationData(message=f"msg {i}"))
        logger.commit(r)

    assert logger.count() == 5
    records = logger.read_all()
    assert len(records) == 5


def test_read_empty(logger):
    assert logger.read_all() == []
    assert logger.count() == 0


def test_survives_corrupted_lines(logger):
    # Write a valid record
    r = logger.begin(SituationData(message="valid"))
    logger.commit(r)

    # Inject a corrupted line
    with open(logger.log_path, "a") as f:
        f.write("not valid json\n")

    # Write another valid record
    r2 = logger.begin(SituationData(message="also valid"))
    logger.commit(r2)

    records = logger.read_all()
    assert len(records) == 2


def test_roundtrip_full_record(logger):
    record = logger.begin(SituationData(
        message="What's the weather?",
        history=["previous msg"],
        metadata={"channel": "slack", "user": "reza"},
    ))
    record.retrieval = RetrievalData(
        candidate_ids=["mem_1", "mem_2", "mem_3"],
        scores=[0.9, 0.5, 0.2],
        selected_indices=[0, 1],
        method="memory_gate",
    )
    record.usage = UsageData(referenced_indices=[0], used_in_response=True)
    record.outcome = OutcomeData(
        signal=OutcomeSignal.POSITIVE,
        source="user_feedback",
        score=0.9,
        details="user said thanks",
    )
    logger.commit(record)

    loaded = logger.read_all()[0]
    assert loaded.situation.message == "What's the weather?"
    assert loaded.retrieval.method == "memory_gate"
    assert loaded.usage.referenced_indices == [0]
    assert loaded.outcome.score == 0.9


def test_extract_training_pairs(logger):
    # Positive outcome with referenced memories
    r = logger.begin(SituationData(message="test"))
    r.retrieval = RetrievalData(
        candidate_ids=["a", "b", "c"],
        scores=[0.9, 0.5, 0.2],
        selected_indices=[0, 1],
    )
    r.retrieval.referenced_indices = [0]  # only "a" was actually used
    r.usage = UsageData(referenced_indices=[0], used_in_response=True)
    r.outcome = OutcomeData(signal=OutcomeSignal.POSITIVE)
    logger.commit(r)

    # Unknown outcome — should be skipped
    r2 = logger.begin(SituationData(message="skip me"))
    logger.commit(r2)

    pairs = logger.extract_training_pairs()
    assert len(pairs) == 1
    situation, pos, neg = pairs[0]
    assert 0 in pos  # "a" is positive
    assert 1 in neg or 2 in neg  # "b" and "c" are negative
