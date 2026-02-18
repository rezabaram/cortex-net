"""Context Assembler — the main pipeline.

Orchestrates all four trainable components to assemble the optimal context
window for a given situation:

    Situation → Situation Encoder → embedding
    embedding → Memory Gate → relevant memories
    embedding → Strategy Selector → strategy profile
    embedding + context → Confidence Estimator → confidence score

    Assembled Context = memories + strategy prompt + confidence framing → LLM
"""
