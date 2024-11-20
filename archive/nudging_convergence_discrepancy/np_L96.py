"""
Code to run and nudge the L96 model using jax and jit.
"""

import numpy as np

ndarray = np.ndarray


class System:
    def __init__(
        self,
        I: int,
        J: int,
        J_sim: int,
        γ1: float,
        γ2: float,
        c1: float,
        c2: float,
        ds: ndarray,
        F: float,
        μ: float,
    ):
        self.I, self.J, self.J_sim = I, J, J_sim

        self.γ1, self.γ2 = γ1, γ2
        self.c1, self.c2 = c1, c2
        self.ds = ds

        self.F = F
        self.μ = μ

        def f(u, v):
            return ode(γ1, γ2, ds, F, u, v)

        self.ode = f

        def step(n, vals):
            (U, V), dt = vals

            u = U[n - 1]
            v = V[n - 1]

            k1u, k1v = f(u, v)
            k2u, k2v = f(u + dt * k1u / 2, v + dt * k1v / 2)
            k3u, k3v = f(u + dt * k2u / 2, v + dt * k2v / 2)
            k4u, k4v = f(u + dt * k3u, v + dt * k3v)

            u1 = u + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
            v1 = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

            U[n] = u1
            V[n] = v1

            return (U, V), dt

        self.step = step


def ode(
    γ1: float,
    γ2: float,
    ds: ndarray,
    F: float,
    u: ndarray,
    v: ndarray,
) -> tuple[ndarray, ndarray]:
    u1 = (
        np.roll(u, 1) * (np.roll(u, -1) - np.roll(u, 2))
        + γ1 * np.sum(u * v.T, axis=0)
        - γ2 * u
        + F
    )

    v1 = -ds * v - np.expand_dims(γ1 * u**2, (1,))

    return u1, v1


def rk4(
    system: System,
    u0: ndarray,
    v0: ndarray,
    t0: float,
    tf: float,
    dt: float,
) -> tuple[ndarray, ndarray]:
    tls = np.arange(t0, tf, dt)
    N = len(tls)

    # Store the solution at every step.
    I, J = system.I, system.J
    U = np.full((N, I), np.inf)
    V = np.full((N, I, J), np.inf)

    # Set initial state.
    U[0] = u0
    V[0] = v0

    for n in range(1, N):
        (U, V), _ = system.step(n, ((U, V), dt))

    return U, V


def interpolate():
    """Fit an interpolation to a solution."""
    raise NotImplementedError()
