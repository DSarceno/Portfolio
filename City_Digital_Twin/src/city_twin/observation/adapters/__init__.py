"""Real observation-source adapters.

Each adapter acquires raw data from an external source and emits the Observation-layer
contract (a :class:`~city_twin.observation.schema.Observation` keyed to the canonical edge
index), so it plugs into the estimator with no downstream changes. See DATASETS.md.
"""
