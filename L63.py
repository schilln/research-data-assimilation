"""
Code to run, nudge, and estimate parameters for the L63 model.
"""

from jax import numpy as jnp

from base_system import System

jndarray = jnp.ndarray


class L63(System):
    def ode(self, true: jndarray) -> jndarray:
        sigma, rho, beta = self.gs

        x, y, z = true

        return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def estimated_ode(
        self,
        cs: jndarray,
        nudged: jndarray,
    ) -> tuple[jndarray, jndarray]:
        sigma, rho, beta = cs

        x, y, z = nudged

        return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
