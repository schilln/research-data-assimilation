"""
Classes to simulate `System`s forward in time.
"""

from jax import numpy as jnp, lax

from base import System

jndarray = jnp.ndarray


class RK4:
    def __init__(self, system: System):
        f = system.f

        def step(i, vals):
            """This function will be jitted."""
            (true, nudged), (dt, cs) = vals
            t = true[i - 1]
            n = nudged[i - 1]

            k1t, k1n = f(cs, t, n)
            k2t, k2n = f(
                cs,
                t + dt * k1t / 2,
                n + dt * k1n / 2,
            )
            k3t, k3n = f(
                cs,
                t + dt * k2t / 2,
                n + dt * k2n / 2,
            )
            k4t, k4n = f(
                cs,
                t + dt * k3t,
                n + dt * k3n,
            )

            t = t.at[:].add((dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t))
            n = n.at[:].add((dt / 6) * (k1n + 2 * k2n + 2 * k3n + k4n))

            true = true.at[i].set(t)
            nudged = nudged.at[i].set(n)

            return (true, nudged), (dt, cs)

        self.step = step

    def solve(
        self,
        system: System,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray, jndarray]:
        tls = jnp.arange(t0, tf, dt)
        N = len(tls)

        # Store the solution at every step.
        true = jnp.full((N, *true0.shape), jnp.inf)
        nudged = jnp.full((N, *nudged0.shape), jnp.inf)

        # Set initial state.
        true = true.at[0].set(true0)
        nudged = nudged.at[0].set(nudged0)

        (true, nudged), _ = lax.fori_loop(
            1, N, self.step, ((true, nudged), (dt, system.cs))
        )

        return true, nudged
