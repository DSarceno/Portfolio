"""State Estimation layer.

Responsibility: estimate the hidden state ``x_t`` and its covariance ``P_t`` from noisy,
partial observations ``(y_t, mask)`` using a Kalman / Extended Kalman Filter. Uses the
dynamics operator from the ``dynamics`` layer; does not define dynamics itself.

Must NOT contain simulation or visualization logic.
"""
