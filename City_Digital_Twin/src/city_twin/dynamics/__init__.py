"""Dynamics / Forecasting layer.

Responsibility: define the state-transition model ``x_{t+dt} = f(x_t, u_t)`` (Phase 1:
graph diffusion + decay; Phase 3: Neural ODE) and roll state forward to produce forecasts
at the 30/60/120-minute horizons, propagating covariance.

Must NOT read raw data directly or render output.
"""
