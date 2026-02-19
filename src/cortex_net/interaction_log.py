"""Interaction Log — structured logging for training data collection.

Captures the full lifecycle of each interaction:
    situation → memories retrieved → memories used → outcome signal

Logs are append-only JSONL files on disk, designed for:
- Training the Memory Gate (which memories were actually useful)
- Training the Strategy Selector (which approach worked)
- Training the Confidence Estimator (predicted vs actual outcome)
- Resumability (logs survive restarts, new entries append)

Schema:
    InteractionRecord {
        id: unique interaction id
        timestamp: ISO 8601
        situation: {message, history, metadata}
        retrieval: {candidates, scores, selected_indices}
        usage: {referenced_indices, used_in_response}
        outcome: {signal, source, details}
        strategy: {name, params}  (future: Phase 3)
        confidence: {predicted, actual}  (future: Phase 4)
    }
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SituationData:
    """The current situation when an interaction occurs."""

    message: str = ""
    history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None  # situation embedding if available


@dataclass
class RetrievalData:
    """What memories were retrieved and how they scored."""

    candidate_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    method: str = "cosine"  # "cosine" | "memory_gate" | "hybrid"


@dataclass
class UsageData:
    """Which retrieved memories were actually used."""

    referenced_indices: list[int] = field(default_factory=list)
    used_in_response: bool = False


class OutcomeSignal:
    """Standardized outcome signal types."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass
class OutcomeData:
    """How the interaction turned out."""

    signal: str = OutcomeSignal.UNKNOWN
    source: str = ""  # "user_feedback" | "implicit" | "self_eval" | "correction"
    score: float | None = None  # 0.0-1.0 graded score if available
    details: str = ""


@dataclass
class InteractionRecord:
    """A single logged interaction."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    situation: SituationData = field(default_factory=SituationData)
    retrieval: RetrievalData = field(default_factory=RetrievalData)
    usage: UsageData = field(default_factory=UsageData)
    outcome: OutcomeData = field(default_factory=OutcomeData)
    # Future phases
    strategy: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionRecord:
        record = cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", ""),
            situation=SituationData(**data.get("situation", {})),
            retrieval=RetrievalData(**data.get("retrieval", {})),
            usage=UsageData(**data.get("usage", {})),
            outcome=OutcomeData(**data.get("outcome", {})),
            strategy=data.get("strategy", {}),
            confidence=data.get("confidence", {}),
        )
        return record


class InteractionLogger:
    """Append-only JSONL logger for interaction records.

    Each interaction is one line in the log file. Writes are atomic
    (append + flush) to minimize data loss on crashes.

    Usage:
        logger = InteractionLogger("./logs")

        # Start logging an interaction
        record = logger.begin(situation=SituationData(message="hello"))

        # Update as the interaction progresses
        record.retrieval = RetrievalData(candidate_ids=["m1", "m2"], scores=[0.9, 0.3])
        record.usage = UsageData(referenced_indices=[0], used_in_response=True)
        record.outcome = OutcomeData(signal="positive", source="user_feedback")

        # Commit to disk
        logger.commit(record)

        # Later: read logs for training
        records = logger.read_all()
    """

    def __init__(self, log_dir: str | Path = "./logs", log_file: str = "interactions.jsonl") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_file

    def begin(self, situation: SituationData | None = None) -> InteractionRecord:
        """Start a new interaction record."""
        record = InteractionRecord()
        if situation is not None:
            record.situation = situation
        return record

    def commit(self, record: InteractionRecord) -> None:
        """Append a completed record to the log file.

        Uses append mode + flush for crash safety.
        """
        line = json.dumps(record.to_dict(), separators=(",", ":")) + "\n"
        with open(self.log_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list[InteractionRecord]:
        """Read all records from the log file."""
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(InteractionRecord.from_dict(data))
                except (json.JSONDecodeError, TypeError):
                    continue  # skip corrupted lines
        return records

    def read_since(self, since_timestamp: str) -> list[InteractionRecord]:
        """Read records after a given ISO timestamp."""
        all_records = self.read_all()
        return [r for r in all_records if r.timestamp >= since_timestamp]

    def count(self) -> int:
        """Count records without loading all into memory."""
        if not self.log_path.exists():
            return 0
        count = 0
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def extract_training_pairs(
        self,
    ) -> list[tuple[SituationData, list[int], list[int]]]:
        """Extract (situation, positive_indices, negative_indices) training pairs.

        A memory is positive if it was referenced/used and the outcome was positive.
        A memory is negative if it was retrieved but not used, or outcome was negative.

        Returns:
            List of (situation, positive_indices, negative_indices) tuples.
        """
        pairs = []
        for record in self.read_all():
            if record.outcome.signal == OutcomeSignal.UNKNOWN:
                continue  # skip unlabeled

            all_indices = set(range(len(record.retrieval.candidate_ids)))
            referenced = set(record.usage.referenced_indices)

            if record.outcome.signal == OutcomeSignal.POSITIVE and referenced:
                positives = list(referenced)
                negatives = list(all_indices - referenced)
                if positives and negatives:
                    pairs.append((record.situation, positives, negatives))
            elif record.outcome.signal == OutcomeSignal.NEGATIVE:
                # If outcome was negative, all selected memories are "wrong"
                selected = set(record.retrieval.selected_indices)
                negatives = list(selected)
                positives_candidates = list(all_indices - selected)
                # We can't know true positives here, skip for now
                # Future: use corrections as positive signal

        return pairs
