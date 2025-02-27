"""
Kuramoto–Sivashinsky Equation
u_t + u_xx + u_xxxx + u u_x = 0, x in an interval [x0, xf] with periodic
boundary conditions.
"""

from functools import partial

import jax

from jax import numpy as jnp
from jax.numpy import fft

from base_system import System

jndarray = jnp.ndarray


class KSE(System):
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
        """

        Parameters
        ----------
        x0, xf
            Endpoints of spatial domain
        xn
            Number of spatial grid points
        """
        super().__init__(μ, gs, bs, cs, observed_slice)

        self._k = fft.rfftfreq(xn, (xf - x0) / xn)

    def ode(self, true: jndarray) -> jndarray:
        # Note `true` should be in frequency domain.
        d = self.d
        p1, p2, p3 = self.gs
        u = true

        return -(
            p1 * d(u, 2)
            + p2 * fft.rfft(fft.irfft(u) * fft.irfft(d(u, 1)))
            + p3 * d(u, 4)
        )

    def estimated_ode(self, cs: jndarray, nudged: jndarray) -> jndarray:
        # Note `nudged` should be in frequency domain.
        d = self.d
        p1, p2, p3 = cs
        u = nudged

        return -(
            p1 * d(u, 2)
            + p2 * fft.rfft(fft.irfft(u) * fft.irfft(d(u, 1)))
            + p3 * d(u, 4)
        )

    def d(self, s: jndarray, m: int) -> jndarray:
        """Compute mth spatial derivative of the state.

        Parameters
        ----------
        s
            System state (e.g., true or nudged) at a point in time in frequency
            domain
        m
            Number of spatial derivatives to take

        Returns
        -------
        d^m s / d {x^m}
            Approximation of mth spatial derivative of s
        """
        return (2 * jnp.pi * 1j * self._k) ** m * s

    @partial(jax.jit, static_argnames="self")
    def _compute_w(self, cs: jndarray, nudged: jndarray) -> jndarray:
        return (
            jax.jacrev(self.estimated_ode, 0, holomorphic=True)(cs, nudged)[
                self.observed_slice
            ].T
            / self.μ
        )
