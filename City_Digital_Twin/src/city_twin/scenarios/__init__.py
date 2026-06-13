"""Scenario Evaluation layer.

Responsibility: define interventions (road closure, accident, rainfall) as inputs ``u_t``
that perturb the network or the transition operator, and compare scenario runs against a
baseline with identical noise seeds.

Must NOT implement the simulation loop itself or mutate estimator internals.
"""
