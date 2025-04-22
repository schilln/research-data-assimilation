"""
One-dimensional wave equation u_tt - c u_xx = 0, x in an interval [x0, xf] with
periodic boundary conditions.

Test case for pseudo-spectral method.

See ACME Volume 4 lab "Spectral 2: A Pseudospectral Method for Periodic
Functions".
"""

from functools import partial

import jax
from jax import numpy as jnp
from jax.numpy import fft

from base_system import System

jndarray = jnp.ndarray


class Wave(System):
    def __init__(
        self,
        μ: float,
        gs: jndarray,
        bs: jndarray,
        cs: jndarray,
        observed_slice: slice,
        x0: float,
        xf: float,
        xn: int,
    ):
        super().__init__(μ, gs, bs, cs, observed_slice)

        self._period = xf - x0
        self.xn = xn

    def ode(self, true: jndarray) -> jndarray:
        u, ut = true
        return jnp.array([ut, self.gs * self.d(u, 2)])

    def estimated_ode(self, cs: jndarray, nudged: jndarray) -> jndarray:
        u, ut = nudged
        return jnp.array([ut, cs * self.d(u, 2)])

    def d(self, s: jndarray, m: int) -> jndarray:
        """Compute mth spatial derivative of the state.

        Parameters
        ----------
        s
            System state (e.g., true or nudged) at a point in time
        m
            Number of spatial derivatives to take

        Returns
        -------
        d^m s / d {x^m}
            Approximation of mth spatial derivative of s
        """
        k = fft.rfftfreq(self.xn, self._period / self.xn)

        return (2 * jnp.pi * 1j * k) ** m * s

    @partial(jax.jit, static_argnames="self")
    def _compute_w(self, cs: jndarray, nudged: jndarray) -> jndarray:
        return (
            jax.jacrev(self.estimated_ode, 0, holomorphic=True)(cs, nudged)[
                self.observed_slice
            ].T
            / self.μ
        )
