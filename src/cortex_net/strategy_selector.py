"""Strategy Selector — learns which approach works for which situation.

Classifies situations into strategy profiles (prompt template + tool
permissions + reasoning style). Learns from which strategy led to
task success in similar past situations.

Architecture: linear classification head + softmax over ~10-20 strategies.
"""
