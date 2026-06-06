"""Small, dependency-light linear algebra utilities.

These helpers are used for stability analysis (e.g. Jacobian spectral radius)
and for educational demonstrations of iterative eigenvalue methods.
"""

from __future__ import annotations

import numpy as np

from simgen.utilities.logging_config import get_logger

logger = get_logger(__name__)


def is_symmetric(matrix: np.ndarray, *, tol: float = 1e-10) -> bool:
    """Return whether ``matrix`` is (numerically) symmetric.

    Parameters
    ----------
    matrix:
        A square 2-D array.
    tol:
        Absolute tolerance for the symmetry check.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {a.shape}")
    return bool(np.allclose(a, a.T, atol=tol))


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the spectral radius (max absolute eigenvalue) of ``matrix``."""
    a = np.asarray(matrix, dtype=float)
    eigenvalues = np.linalg.eigvals(a)
    return float(np.max(np.abs(eigenvalues)))


def condition_number(matrix: np.ndarray) -> float:
    """Return the 2-norm condition number of ``matrix``.

    A large condition number indicates an ill-conditioned problem where small
    perturbations in the input can produce large changes in the output.
    """
    a = np.asarray(matrix, dtype=float)
    return float(np.linalg.cond(a))


def power_iteration(
    matrix: np.ndarray,
    *,
    num_iterations: int = 1000,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Estimate the dominant eigenpair of ``matrix`` via power iteration.

    Parameters
    ----------
    matrix:
        Square matrix.
    num_iterations:
        Maximum number of iterations.
    tol:
        Convergence tolerance on the eigenvalue estimate.

    Returns
    -------
    tuple[float, numpy.ndarray]
        ``(eigenvalue, eigenvector)`` for the dominant eigenvalue. The
        eigenvector is normalised to unit length.

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {a.shape}")

    n = a.shape[0]
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(n)
    vector /= np.linalg.norm(vector)

    eigenvalue = 0.0
    for _ in range(num_iterations):
        product = a @ vector
        norm = np.linalg.norm(product)
        if norm == 0.0:
            logger.warning("Power iteration collapsed to the zero vector")
            return 0.0, vector
        new_vector = product / norm
        new_eigenvalue = float(new_vector @ (a @ new_vector))
        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, new_vector
        eigenvalue, vector = new_eigenvalue, new_vector

    logger.debug("Power iteration did not converge within %d iterations", num_iterations)
    return eigenvalue, vector
