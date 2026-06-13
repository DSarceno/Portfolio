"""Simulation layer.

Responsibility: run the closed loop forward over time, alternating dynamics prediction and
(when observations are available) estimator updates, producing state trajectories.

Must NOT embed scenario/intervention policy or API code.
"""
