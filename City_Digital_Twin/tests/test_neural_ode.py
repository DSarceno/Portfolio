"""Phase-3 validation: the hybrid Neural ODE vs. the linear baseline.

These tests train a small Neural ODE on trajectories from a *nonlinear* ground-truth
process (logistic congestion growth) that the linear graph-diffusion model structurally
cannot represent, and check that:

* the model conforms to the dynamics interface and drives the Kalman Filter as an EKF;
* its forecast RMSE at 30/60/120 min beats the linear baseline.

torch is an optional dependency; tests skip cleanly if it is not installed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdiffeq")

from city_twin.dynamics.base import DynamicsModel  # noqa: E402
from city_twin.dynamics.graph_diffusion import GraphDiffusionDynamics  # noqa: E402
from city_twin.dynamics.neural_ode import NeuralODEDynamics, train_neural_ode  # noqa: E402
from city_twin.estimation.kalman import GaussianState, KalmanFilter  # noqa: E402
from city_twin.ingestion.network import RoadNetwork  # noqa: E402
from city_twin.observation.synthetic import SyntheticObservations  # noqa: E402

DECAY = 0.1
DIFFUSION = 0.05
GROWTH = 0.15  # nonlinear logistic term the linear model cannot capture


def _network() -> RoadNetwork:
    graph = nx.DiGraph()
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)]:
        graph.add_edge(u, v, length=400.0, free_flow_speed=60.0)
    return RoadNetwork.from_digraph(graph)


def _nonlinear_trajectory(net: RoadNetwork, x0: np.ndarray, n_steps: int) -> np.ndarray:
    """Euler rollout of dx/dt = -decay x - diffusion L x + growth x(1-x), clipped."""
    laplacian = net.laplacian()
    x = np.clip(x0.astype(float), 0.0, 1.0)
    traj = [x.copy()]
    for _ in range(n_steps):
        dx = -DECAY * x - DIFFUSION * (laplacian @ x) + GROWTH * x * (1.0 - x)
        x = np.clip(x + dx, 0.0, 1.0)
        traj.append(x.copy())
    return np.array(traj)


@pytest.fixture(scope="module")
def trained():
    """Train one Neural ODE on nonlinear trajectories; reused across tests."""
    net = _network()
    rng = np.random.default_rng(0)
    train_trajs = np.array(
        [_nonlinear_trajectory(net, rng.uniform(0.05, 0.6, net.n_edges), 20) for _ in range(12)]
    )
    model = NeuralODEDynamics(net.laplacian(), net.adjacency, hidden=16, seed=0)
    final_loss = train_neural_ode(model, train_trajs, epochs=500, lr=0.02, seed=0)
    linear = GraphDiffusionDynamics(
        laplacian=net.laplacian(), decay=DECAY, diffusion=DIFFUSION, dt_minutes=5.0
    )
    return net, model, linear, final_loss


def test_conforms_to_dynamics_interface(trained) -> None:
    _net, model, _linear, _loss = trained
    assert isinstance(model, DynamicsModel)
    assert model.steps_for_minutes(120) == 24


def test_predict_shapes_and_symmetry(trained) -> None:
    net, model, _linear, _loss = trained
    n = net.n_edges
    mean, cov = model.predict(np.full(n, 0.3), np.eye(n) * 0.1, np.eye(n) * 1e-3)
    assert mean.shape == (n,)
    assert cov.shape == (n, n)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)


def test_training_reduced_loss(trained) -> None:
    _net, _model, _linear, final_loss = trained
    assert final_loss < 1e-3  # fits one-step transitions well


def test_neural_ode_beats_linear_on_forecast(trained) -> None:
    net, model, linear, _loss = trained
    n = net.n_edges
    x0 = np.full(n, 0.25)
    truth = _nonlinear_trajectory(net, x0, 24)  # held-out continuation

    q = np.eye(n) * 1e-4
    cov0 = np.eye(n) * 1e-3
    node_means, _ = model.forecast(x0, cov0, q, 24)
    lin_means, _ = linear.forecast(x0, cov0, q, 24)

    node_rmse = np.sqrt(np.mean((node_means - truth[1:]) ** 2))
    lin_rmse = np.sqrt(np.mean((lin_means - truth[1:]) ** 2))
    assert node_rmse < lin_rmse  # the learned residual captures the nonlinearity

    # Sanity at each required horizon: Neural ODE no worse than linear.
    for minutes in (30, 60, 120):
        k = model.steps_for_minutes(minutes) - 1
        assert abs(node_means[k] - truth[k + 1]).mean() <= abs(lin_means[k] - truth[k + 1]).mean()


def test_neural_ode_drives_ekf(trained) -> None:
    net, model, _linear, _loss = trained
    n = net.n_edges
    truth = _nonlinear_trajectory(net, np.full(n, 0.2), 30)
    synth = SyntheticObservations(net, model, noise_speed_kmh=4.0, observed_fraction=0.6, seed=7)
    observations = synth.observe(truth)

    kf = KalmanFilter(model, process_cov=np.eye(n) * 1e-3, measurement_var=synth.measurement_var())
    initial = GaussianState(mean=np.full(n, 0.1), cov=np.eye(n) * 1e-2)
    beliefs = kf.filter(initial, observations)

    est = np.array([b.mean for b in beliefs])
    assert np.all(np.isfinite(est))
    rmse = np.sqrt(np.mean((est - truth) ** 2))
    assert rmse < 0.2  # EKF tracks the hidden nonlinear state
