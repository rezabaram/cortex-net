"""State Manager — persistent checkpointing for cortex-net components.

Handles saving and loading of model weights, optimizer state, and training
metadata. All writes are atomic (temp file + rename) to prevent corruption.
State format is versioned for forward compatibility.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


STATE_FORMAT_VERSION = 1


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside each checkpoint."""

    format_version: int = STATE_FORMAT_VERSION
    component_name: str = ""
    epoch: int = 0
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class StateManager:
    """Manages persistent state for trainable components.

    Features:
    - Atomic writes (write to temp, rename into place)
    - Auto-resume from latest checkpoint
    - Configurable checkpoint retention (max_checkpoints)
    - Versioned format for forward compatibility

    Usage:
        sm = StateManager("./state")
        sm.save(model, optimizer, CheckpointMetadata(component_name="memory_gate", epoch=5))
        sm.load(model, optimizer, component_name="memory_gate")
    """

    def __init__(self, state_dir: str | Path = "./state", max_checkpoints: int = 5) -> None:
        self.state_dir = Path(state_dir)
        self.max_checkpoints = max_checkpoints
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _component_dir(self, component_name: str) -> Path:
        d = self.state_dir / component_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _checkpoint_path(self, component_name: str, step: int) -> Path:
        return self._component_dir(component_name) / f"checkpoint_{step:08d}.pt"

    def _metadata_path(self, checkpoint_path: Path) -> Path:
        return checkpoint_path.with_suffix(".json")

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        metadata: CheckpointMetadata,
    ) -> Path:
        """Save a checkpoint atomically.

        Returns the path to the saved checkpoint.
        """
        component_dir = self._component_dir(metadata.component_name)
        ckpt_path = self._checkpoint_path(metadata.component_name, metadata.step)

        payload: dict[str, Any] = {
            "format_version": STATE_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "metadata": asdict(metadata),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        # Atomic write: save to temp file in same dir, then rename
        fd, tmp_path = tempfile.mkstemp(dir=component_dir, suffix=".tmp")
        try:
            os.close(fd)
            torch.save(payload, tmp_path)
            os.replace(tmp_path, ckpt_path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Also write human-readable metadata
        meta_path = self._metadata_path(ckpt_path)
        fd2, tmp_meta = tempfile.mkstemp(dir=component_dir, suffix=".tmp")
        try:
            os.close(fd2)
            with open(tmp_meta, "w") as f:
                json.dump(asdict(metadata), f, indent=2)
            os.replace(tmp_meta, meta_path)
        except BaseException:
            try:
                os.unlink(tmp_meta)
            except OSError:
                pass
            raise

        self._prune_old_checkpoints(metadata.component_name)
        return ckpt_path

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        component_name: str = "",
        checkpoint_path: Path | None = None,
    ) -> CheckpointMetadata | None:
        """Load the latest (or specified) checkpoint.

        Returns metadata if a checkpoint was loaded, None if no checkpoint exists.
        """
        if checkpoint_path is None:
            checkpoint_path = self.latest_checkpoint(component_name)
        if checkpoint_path is None:
            return None

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Version check
        fmt_ver = payload.get("format_version", 0)
        if fmt_ver > STATE_FORMAT_VERSION:
            raise ValueError(
                f"Checkpoint format version {fmt_ver} is newer than supported {STATE_FORMAT_VERSION}"
            )

        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        meta_dict = payload.get("metadata", {})
        return CheckpointMetadata(**meta_dict)

    def latest_checkpoint(self, component_name: str) -> Path | None:
        """Find the latest checkpoint for a component."""
        component_dir = self._component_dir(component_name)
        checkpoints = sorted(component_dir.glob("checkpoint_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def list_checkpoints(self, component_name: str) -> list[Path]:
        """List all checkpoints for a component, oldest first."""
        component_dir = self._component_dir(component_name)
        return sorted(component_dir.glob("checkpoint_*.pt"))

    def _prune_old_checkpoints(self, component_name: str) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        if self.max_checkpoints <= 0:
            return
        checkpoints = self.list_checkpoints(component_name)
        while len(checkpoints) > self.max_checkpoints:
            old = checkpoints.pop(0)
            old.unlink(missing_ok=True)
            self._metadata_path(old).unlink(missing_ok=True)
