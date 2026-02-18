"""Confidence Estimator — predicts how confident the system should be.

Trained on calibration against actual outcomes. When it says 70% confident,
it should be right ~70% of the time. Low confidence triggers hedging,
clarification requests, or escalation.

Architecture: 2-layer MLP → scalar output [0, 1].
Training: calibration loss (not just accuracy).
"""
