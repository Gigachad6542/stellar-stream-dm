"""Neural network models for stellar stream classification and embedding.

Modules:
    gnn        — GINEConv encoder, classifier, and multi-task model
    baseline   — 1D CNN baseline on binned density profiles
    transformer — Transformer encoder (optional ensemble member)
    utils      — Checkpoint I/O, normalizer persistence, LR scheduling
"""
