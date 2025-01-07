"""
Code to run, nudge, and estimate parameters for the L63 model.
"""

from jax import numpy as jnp

from base_system import System

jndarray = jnp.ndarray


class L63(System):
    def ode(self, true: jndarray) -> jndarray:
        σ, ρ, β = self.γs

        x, y, z = true

        return jnp.array([σ * (y - x), x * (ρ - z) - y, x * y - β * z])

    def estimated_ode(
        self,
        cs: jndarray,
        nudged: jndarray,
    ) -> tuple[jndarray, jndarray]:
        σ, ρ, β = cs

        x, y, z = nudged

        return jnp.array([σ * (y - x), x * (ρ - z) - y, x * y - β * z])
