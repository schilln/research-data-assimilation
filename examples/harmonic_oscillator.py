"""
Code to run, nudge, and estimate parameters for the L63 model.
"""

from jax import numpy as jnp

from base_system import System

jndarray = jnp.ndarray


class HarmonicOscillator(System):
    def ode(self, true: jndarray) -> jndarray:
        a, b, c, d = self.gs

        x, y = true

        return jnp.array([a * y, b * x + c * y * jnp.cos(d * x)])

    def estimated_ode(self, cs: jndarray, nudged: jndarray) -> jndarray:
        b0, a1, c1, d1, a0, b1, c0, d0, e0, e1, f0, f1, g0, g1 = cs

        x, y = nudged

        return jnp.array(
            [
                a0 * x
                + b0 * y
                # + c0 * y * jnp.cos(d0 * x)
                # + e0 * x * jnp.cos(f0 * y)
                + g0,
                a1 * x
                + b1 * y
                + c1 * y * jnp.cos(d1 * x)
                # + e1 * x * jnp.cos(f1 * y)
                + g1,
            ]
        )
