"""
Code to run, nudge, and estimate parameters for the L63 model.
"""

from jax import numpy as jnp

from base_system import System

jndarray = jnp.ndarray


class L63(System):
    def __init__(self, *args, modified: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        if modified:
            # Use `modified_estimated_ode` which allows extra "fake" parameters.
            self.estimated_ode = self.modified_estimated_ode

    def ode(self, true: jndarray) -> jndarray:
        sigma, rho, beta = self.gs

        x, y, z = true

        return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def estimated_ode(self, cs: jndarray, nudged: jndarray) -> jndarray:
        sigma, rho, beta = cs

        x, y, z = nudged

        return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def modified_estimated_ode(
        self, cs: jndarray, nudged: jndarray
    ) -> jndarray:
        sigma, rho, beta, *a = cs

        x, y, z = nudged

        return jnp.array(
            [
                sigma * (y - x) + a[0],
                x * (rho - z) - y,
                x * y - beta * z + y * a[1],
            ]
        )
