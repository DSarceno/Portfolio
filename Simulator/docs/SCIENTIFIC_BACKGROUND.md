# Scientific Background

This document covers the science behind the framework and its reference
phenomena: the modelling formalism, the numerical methods, the diagnostics, and
the five built-in systems.

## 1. Modelling formalism: first-order ODE systems

Every phenomenon is expressed as a first-order system of ordinary differential
equations,
$$ \frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}), \qquad \mathbf{y}(t_0) = \mathbf{y}_0, $$
with $\mathbf{y} \in \mathbb{R}^n$. This is fully general:

- A second-order equation $\ddot{x} = a(t, x, \dot{x})$ becomes the pair
  $\dot{x} = v,\ \dot{v} = a(t, x, v)$.
- A spatial PDE is reduced by the **method of lines**: discretise space (e.g.
  finite differences) to obtain a large system of ODEs in time.

## 2. Numerical integration

### Explicit Euler
$$ \mathbf{y}_{n+1} = \mathbf{y}_n + h\,\mathbf{f}(t_n, \mathbf{y}_n). $$
First-order accurate ($O(h)$ global error); only conditionally stable. Included
for pedagogy.

### Classical Runge–Kutta (RK4)
$$
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(t_n, \mathbf{y}_n), \\
\mathbf{k}_2 &= \mathbf{f}(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_1), \\
\mathbf{k}_3 &= \mathbf{f}(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_2), \\
\mathbf{k}_4 &= \mathbf{f}(t_n + h, \mathbf{y}_n + h\,\mathbf{k}_3), \\
\mathbf{y}_{n+1} &= \mathbf{y}_n + \tfrac{h}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4).
\end{aligned}
$$
Fourth-order accurate ($O(h^4)$ global error); the default for smooth, non-stiff
systems.

### Adaptive and stiff methods
The framework bridges to SciPy's `solve_ivp`: `RK45`/`DOP853` (adaptive,
non-stiff), and `Radau`/`BDF`/`LSODA` (implicit, for stiff problems where
explicit methods require impractically small steps).

### Symplectic integration
For conservative mechanical systems, velocity Verlet conserves a shadow
Hamiltonian and exhibits no secular energy drift, unlike RK4.

## 3. Diagnostics

- **Power spectrum / dominant frequency** (FFT): identify oscillation
  frequencies.
- **Autocorrelation**: characterise periodicity and decorrelation time.
- **Relative energy drift** $\max_t |E(t)-E_0|/|E_0|$: validates conservative
  integrations.
- **Largest Lyapunov exponent** (Rosenstein's nearest-neighbour method):
  quantifies sensitive dependence on initial conditions; a positive value is the
  defining signature of chaos.

## 4. Reference phenomena

### 4.1 Simple harmonic oscillator
$$ \dot{x} = v, \qquad \dot{v} = -\omega^2 x. $$
Solution $x(t) = A\cos(\omega t + \phi)$; conserved energy
$E = \tfrac12 v^2 + \tfrac12\omega^2 x^2$. Phase-space orbits are ellipses. Used
as the analytic-accuracy and energy-conservation benchmark.

### 4.2 Lorenz system
$$ \dot{x} = \sigma(y-x), \quad \dot{y} = x(\rho-z)-y, \quad \dot{z} = xy-\beta z. $$
With $\sigma=10,\ \rho=28,\ \beta=8/3$ the system is chaotic with a strange
attractor of fractal dimension $\approx 2.06$ and largest Lyapunov exponent
$\approx 0.906$. Derived from a Galerkin truncation of Rayleigh–Bénard
convection.

### 4.3 Double pendulum
Two coupled rigid pendulums; derived from the Euler–Lagrange equations. Energy
is conserved by the exact flow,
$$ E = \tfrac12 m_1 (l_1\dot\theta_1)^2 + \tfrac12 m_2\big[(l_1\dot\theta_1)^2 + (l_2\dot\theta_2)^2 + 2 l_1 l_2 \dot\theta_1\dot\theta_2\cos(\theta_1-\theta_2)\big] - (m_1+m_2)g l_1\cos\theta_1 - m_2 g l_2\cos\theta_2, $$
and the motion is chaotic for generic energies. (Note: for a horizontal start
$E_0\approx 0$, so the *relative* energy-drift metric is ill-conditioned;
absolute drift is the meaningful quantity there.)

### 4.4 Duffing oscillator
$$ \ddot{x} + \delta\dot{x} + \alpha x + \beta x^3 = \gamma\cos(\omega t). $$
A damped, driven oscillator with cubic stiffness. The double-well case
($\alpha<0,\ \beta>0$) exhibits homoclinic tangling and a chaotic attractor.

### 4.5 Van der Pol oscillator
$$ \ddot{x} - \mu(1-x^2)\dot{x} + x = 0. $$
Nonlinear damping injects energy at small amplitude and dissipates it at large
amplitude, producing a unique attracting **limit cycle** (Poincaré–Bendixson).
For large $\mu$ the motion becomes a relaxation oscillation.

## 5. Validity and limitations

Numerical solutions approximate the exact flow up to truncation and round-off
error. For chaotic systems, individual trajectories are predictable only over a
horizon $\sim 1/\lambda_{\max}$; statistical and geometric properties (attractor
shape, invariant measures, Lyapunov spectrum) remain robust. Method choice
(explicit vs implicit, step size) must match the system's stiffness and the
required accuracy.

## References

See [`../reports/bibliography.bib`](../reports/bibliography.bib) for full
citations (Strogatz 2015; Lorenz 1963; Guckenheimer & Holmes 1983; Hairer,
Nørsett & Wanner 1993; Rosenstein et al. 1993; and others).
