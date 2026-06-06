"""Matplotlib rendering of simulation results.

Figures are produced from a :class:`PlotSpec` plus a
:class:`~simgen.simulations.base.SimulationResult`. The module uses the
non-interactive ``Agg`` backend so it works in headless environments and never
opens a window; figures are always saved to disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401 (registers 3d projection)

from simgen.simulations.base import SimulationResult  # noqa: E402
from simgen.simulations.spec import PhenomenonSpec, PlotSpec  # noqa: E402
from simgen.utilities.io import ensure_directory  # noqa: E402
from simgen.utilities.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def plot_result(
    plot: PlotSpec,
    result: SimulationResult,
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Render a single :class:`PlotSpec` to an image file.

    Parameters
    ----------
    plot:
        The plot specification (kind + axis variables).
    result:
        The simulation trajectory to visualise.
    output_path:
        Destination image path (extension determines the format).
    dpi:
        Output resolution.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    KeyError
        If a referenced variable is not present in ``result``.
    ValueError
        If the plot kind is unsupported.
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    if plot.kind == "time_series":
        fig = _plot_time_series(plot, result)
    elif plot.kind == "phase2d":
        fig = _plot_phase2d(plot, result)
    elif plot.kind == "phase3d":
        fig = _plot_phase3d(plot, result)
    else:  # pragma: no cover - guarded by PlotSpec validation
        raise ValueError(f"Unsupported plot kind: {plot.kind!r}")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot %s", output_path)
    return output_path


def _label(result: SimulationResult, symbol: str) -> str:
    """Return an axis label for a symbol (``t`` becomes ``time``)."""
    return "time" if symbol == "t" else symbol


def _plot_time_series(plot: PlotSpec, result: SimulationResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_symbol = plot.y or result.symbols[0]
    ax.plot(result.t, result.column(y_symbol), lw=1.2, color="#1f77b4")
    ax.set_xlabel("time")
    ax.set_ylabel(y_symbol)
    ax.set_title(plot.title or f"{y_symbol}(t)")
    ax.grid(True, alpha=0.3)
    return fig


def _plot_phase2d(plot: PlotSpec, result: SimulationResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_symbol = plot.x
    y_symbol = plot.y or result.symbols[min(1, len(result.symbols) - 1)]
    ax.plot(result.column(x_symbol), result.column(y_symbol), lw=0.8, color="#d62728")
    ax.set_xlabel(_label(result, x_symbol))
    ax.set_ylabel(_label(result, y_symbol))
    ax.set_title(plot.title or f"{y_symbol} vs {x_symbol}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")
    return fig


def _plot_phase3d(plot: PlotSpec, result: SimulationResult) -> plt.Figure:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    x_symbol, y_symbol, z_symbol = plot.x, plot.y, plot.z
    if y_symbol is None or z_symbol is None:
        raise ValueError("phase3d plot requires x, y and z symbols")
    ax.plot(
        result.column(x_symbol),
        result.column(y_symbol),
        result.column(z_symbol),
        lw=0.5,
        color="#2ca02c",
    )
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(y_symbol)
    ax.set_zlabel(z_symbol)
    ax.set_title(plot.title or f"{x_symbol}-{y_symbol}-{z_symbol}")
    return fig


def render_all_plots(
    spec: PhenomenonSpec,
    result: SimulationResult,
    output_dir: str | Path,
    *,
    prefix: str = "figure",
    dpi: int = 150,
) -> list[Path]:
    """Render every plot in ``spec`` and return the written paths.

    Parameters
    ----------
    spec:
        The phenomenon specification (its ``plots`` list drives output).
    result:
        The trajectory to visualise.
    output_dir:
        Directory for the generated images.
    prefix:
        Filename prefix; files are ``<prefix>_<index>_<kind>.png``.
    dpi:
        Output resolution.

    Returns
    -------
    list[pathlib.Path]
        Paths of the written figures.
    """
    out = ensure_directory(output_dir)
    written: list[Path] = []
    for index, plot in enumerate(spec.plots):
        path = out / f"{prefix}_{index}_{plot.kind}.png"
        written.append(plot_result(plot, result, path, dpi=dpi))
    return written
