"""Tests for time-series and dynamical-systems analysis."""

from __future__ import annotations

import numpy as np
import pytest

from simgen import run_spec
from simgen.mathematics import (
    autocorrelation,
    dominant_frequency,
    largest_lyapunov_exponent,
    power_spectrum,
    relative_energy_drift,
)
from simgen.scenes.components import sample_trajectory
from simgen.simulations.library import get_spec
from simgen.simulations.runner import energy_series


def test_power_spectrum_peak_at_signal_frequency() -> None:
    """The spectral peak should sit at the input sinusoid's frequency."""
    freq = 3.0
    dt = 0.001
    t = np.arange(0, 4, dt)
    signal = np.sin(2 * np.pi * freq * t)
    freqs, power = power_spectrum(signal, dt)
    assert freqs[np.argmax(power)] == pytest.approx(freq, abs=0.5)


def test_dominant_frequency() -> None:
    dt = 0.001
    t = np.arange(0, 4, dt)
    signal = np.cos(2 * np.pi * 5.0 * t)
    assert dominant_frequency(signal, dt) == pytest.approx(5.0, abs=0.5)


def test_autocorrelation_zero_lag_is_one() -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(500)
    acf = autocorrelation(signal, max_lag=10)
    assert acf[0] == pytest.approx(1.0)
    assert acf.shape == (11,)


def test_relative_energy_drift_constant_is_zero() -> None:
    energy = np.full(100, 7.0)
    assert relative_energy_drift(energy) == pytest.approx(0.0)


def test_harmonic_oscillator_conserves_energy() -> None:
    """RK4 keeps the SHO energy nearly constant over the run."""
    spec = get_spec("simple-harmonic-oscillator")
    result = run_spec(spec)
    energy = energy_series(spec, result)
    assert energy is not None
    assert relative_energy_drift(energy) < 1e-3


def test_lyapunov_chaotic_exceeds_regular() -> None:
    """Lorenz (chaotic) has a clearly larger LLE than the SHO (regular)."""
    lorenz = run_spec(get_spec("lorenz-system"))
    sampled = sample_trajectory(lorenz.states, max_points=3000)
    eff_dt = (lorenz.t[-1] - lorenz.t[0]) / (sampled.shape[0] - 1)
    lle_chaos = largest_lyapunov_exponent(sampled, eff_dt, fit_fraction=0.5)

    sho = run_spec(get_spec("simple-harmonic-oscillator"))
    lle_regular = largest_lyapunov_exponent(sho.states, sho.dt, fit_fraction=0.5)

    assert lle_chaos > 0.4
    assert lle_chaos > lle_regular
