"""
Code to run and nudge the L96 model using jax and jit.
"""

from collections.abc import Callable

import jax
from jax import numpy as jnp, jit, lax
# jax.config.update("jax_enable_x64", True)

# Useful: jax.debug.print

jndarray = jnp.ndarray


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
        ds: jndarray,
        F: float,
        μ: float,
    ):
        self.I, self.J, self.J_sim = I, J, J_sim

        self.γ1, self.γ2 = γ1, γ2
        self.c1, self.c2 = c1, c2
        self.ds = ds

        self.F = F
        self.μ = μ

    def get_true_params(self) -> tuple:
        return (self.I, self.J, self.γ1, self.γ2, self.ds, self.F)

    def get_nudged_params(self) -> tuple:
        return (self.I, self.J_sim, self.c1, self.c2, self.ds, self.F, self.μ)


def ode(
    params: tuple,
    u: jndarray,
    v: jndarray,
) -> tuple[jndarray, jndarray]:
    I, J, γ1, γ2, ds, F = params

    u1 = (
        jnp.roll(u, 1) * (jnp.roll(u, -1) - jnp.roll(u, 2))
        + γ1 * jnp.sum(u * v.T, axis=0)
        - γ2 * u
        + F
    )

    v1 = -ds * v - jnp.reshape(γ1 * u**2, (I, 1))

    return u1, v1


def rk4(
    ode: Callable[[tuple, jndarray, jndarray], tuple[jndarray, jndarray]],
    params: tuple,
    u0: jndarray,
    v0: jndarray,
    t0: float,
    tf: float,
    dt: float,
    # true_sol: None | Callable[[float], jndarray],
) -> Callable[[float], jndarray]:
    f = lambda u, v: ode(params, u, v)
    I, J, *_ = params

    tls = jnp.arange(t0, tf, dt)
    N = len(tls)

    # Store the solution at every step.
    U = jnp.full((N, I), jnp.inf)
    V = jnp.full((N, I, J), jnp.inf)

    # Set initial state.
    U = U.at[0].set(u0)
    V = V.at[0].set(v0)

    for n in range(1, N):
        u = U[n - 1]
        v = V[n - 1]

        k1u, k1v = f(u, v)
        k2u, k2v = f(u + dt * k1u / 2, v + dt * k1v / 2)
        k3u, k3v = f(u + dt * k2u / 2, v + dt * k2v / 2)
        k4u, k4v = f(u + dt * k3u, v + dt * k3v)

        u1 = u + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
        v1 = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

        U = U.at[n].set(u1)
        V = V.at[n].set(v1)

    return U, V


def interpolate():
    """Fit an interpolation to a solution."""
    raise NotImplementedError()
