"""Quantitative analysis of simulation time series.

These functions turn raw trajectories into interpretable diagnostics: spectral
content, autocorrelation, energy conservation drift and a finite-time estimate
of the largest Lyapunov exponent (a hallmark of chaos).
"""

from __future__ import annotations

import numpy as np

from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)


def power_spectrum(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided power spectrum of a real signal.

    Parameters
    ----------
    signal:
        1-D real-valued time series.
    dt:
        Sampling interval (time between samples).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(frequencies, power)`` for non-negative frequencies.
    """
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    n = x.shape[0]
    if n < 2:
        raise ValueError("signal must contain at least two samples")
    spectrum = np.fft.rfft(x)
    power = (np.abs(spectrum) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, power


def dominant_frequency(signal: np.ndarray, dt: float) -> float:
    """Return the frequency carrying the most power (ignoring the DC term)."""
    freqs, power = power_spectrum(signal, dt)
    if freqs.shape[0] <= 1:
        return 0.0
    idx = int(np.argmax(power[1:])) + 1
    return float(freqs[idx])


def autocorrelation(signal: np.ndarray, *, max_lag: int | None = None) -> np.ndarray:
    """Compute the normalised autocorrelation function of ``signal``.

    Parameters
    ----------
    signal:
        1-D time series.
    max_lag:
        Maximum lag to return. Defaults to ``len(signal) - 1``.

    Returns
    -------
    numpy.ndarray
        Autocorrelation values for lags ``0 .. max_lag`` (``r[0] == 1``).
    """
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    n = x.shape[0]
    if n < 2:
        raise ValueError("signal must contain at least two samples")
    full = np.correlate(x, x, mode="full")
    mid = full.size // 2
    acf = full[mid:]
    denom = acf[0] if acf[0] != 0 else 1.0
    acf = acf / denom
    if max_lag is None:
        return acf
    return acf[: max_lag + 1]


def relative_energy_drift(energy: np.ndarray) -> float:
    """Return the relative drift in an energy time series.

    Defined as ``max|E(t) - E(0)| / |E(0)|``. A small value indicates good
    energy conservation, which is an important validity check for conservative
    mechanical simulations.
    """
    e = np.asarray(energy, dtype=float)
    if e.shape[0] == 0:
        raise ValueError("energy array is empty")
    e0 = e[0]
    if e0 == 0.0:
        return float(np.max(np.abs(e - e0)))
    return float(np.max(np.abs(e - e0)) / abs(e0))


def largest_lyapunov_exponent(
    trajectory: np.ndarray,
    dt: float,
    *,
    horizon: int | None = None,
    min_separation: float = 1e-12,
    fit_fraction: float = 1.0,
) -> float:
    """Estimate the largest Lyapunov exponent (Rosenstein's algorithm).

    For each point on the trajectory the nearest spatial neighbour is found
    (excluding temporally close points via a Theiler window). The mean
    logarithmic divergence of these neighbour pairs is tracked as a function of
    forward time; its slope (per unit time) estimates the largest Lyapunov
    exponent. A clearly positive value is a signature of chaos.

    Parameters
    ----------
    trajectory:
        State history of shape ``(n_steps, n_dim)``.
    dt:
        Time between samples.
    horizon:
        Number of forward steps over which divergence is tracked. Defaults to a
        value scaled to the trajectory length.
    min_separation:
        Distances below this are ignored (avoids ``log(0)``).
    fit_fraction:
        Fraction of the horizon (from the start) used for the linear slope fit;
        useful to exclude the saturation plateau.

    Returns
    -------
    float
        Estimated largest Lyapunov exponent (per unit time).

    Raises
    ------
    ValueError
        If the trajectory is too short.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim == 1:
        traj = traj[:, None]
    n = traj.shape[0]
    if n < 50:
        raise ValueError("trajectory too short for a Lyapunov estimate")

    theiler = max(1, n // 50)
    if horizon is None:
        horizon = min(max(20, n // 20), 400)

    # Nearest spatial neighbour (outside the Theiler window) for each point.
    neighbours = np.full(n, -1, dtype=int)
    for i in range(n):
        diffs = traj - traj[i]
        dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        dists[lo:hi] = np.inf
        j = int(np.argmin(dists))
        if np.isfinite(dists[j]):
            neighbours[i] = j

    # Mean log-divergence at each forward lag.
    log_sum = np.zeros(horizon)
    counts = np.zeros(horizon)
    for i in range(n):
        j = neighbours[i]
        if j < 0:
            continue
        max_k = min(horizon, n - i, n - j)
        if max_k <= 0:
            continue
        seg = traj[i : i + max_k] - traj[j : j + max_k]
        dist = np.sqrt(np.einsum("ij,ij->i", seg, seg))
        valid = dist > min_separation
        log_sum[:max_k][valid] += np.log(dist[valid])
        counts[:max_k][valid] += 1

    mask = counts > 0
    if mask.sum() < 2:
        logger.warning("Lyapunov estimate found too few neighbour pairs")
        return 0.0
    divergence = log_sum[mask] / counts[mask]
    times = np.nonzero(mask)[0] * dt

    fit_n = max(2, int(len(times) * fit_fraction))
    slope = np.polyfit(times[:fit_n], divergence[:fit_n], 1)[0]
    return float(slope)
