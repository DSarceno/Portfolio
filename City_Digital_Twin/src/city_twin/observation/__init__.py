"""Observation layer.

Responsibility: define the observation schema and the measurement/noise model; emit noisy,
partial observations ``y_t`` plus an observation mask, keyed to the canonical edge index.
In Phase 1 this includes the synthetic observation generator.

Must NOT contain state-estimation or dynamics logic. Depends on the ingestion graph.
"""
