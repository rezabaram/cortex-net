"""Memory Gate — learned relevance scoring for memory retrieval.

Replaces naive cosine similarity with a trained bilinear scorer that learns
which memories are actually relevant given the current situation.

Architecture: bilinear scoring function
    relevance(situation, memory) = situation · W · memory

Training signal: did retrieving this memory lead to a better outcome?
"""
