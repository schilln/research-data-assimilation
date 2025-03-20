"""
This is my attempt to code 3.1. Harmonic Oscillator — Fourier Frequency into this setup
"""

from jax import numpy as jnp

from base_system import System

jndarray = jnp.ndarray


class FF(System):
    def ode(self, true: jndarray) -> jndarray:
        a, b, c, d = self.gs

        x, y = true

        return jnp.array([a*y, -b*x + c*y*jnp.cos(d*x)])

    def estimated_ode(
        self,
        cs: jndarray,
        nudged: jndarray,
    ) -> tuple[jndarray, jndarray]:
        a, b, c, d = cs

        x, y = nudged

        return jnp.array([a*y, -b*x + c*y*jnp.cos(d*x)])
