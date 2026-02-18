"""Situation Encoder — builds a dense vector representation of the current situation.

Takes the current message, recent history, and metadata, and produces a
situation embedding that all other components consume.

Architecture: 2-layer transformer or MLP.
Output: situation embedding vector (e.g., 256-dim).
"""
