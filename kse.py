"""
Kuramoto–Sivashinsky Equation
u_t + u_xx + u_xxxx + u u_x = 0, x in an interval [x0, xf] with periodic
boundary conditions.
"""

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
    ):
        super().__init__(μ, gs, bs, cs, observed_slice)

        self._period = xf - x0

    def ode(self, true: jndarray) -> jndarray:
        d = self.d
        p1, p2, p3 = self.gs
        u = true

        return -(p1 * d(u, 2) + p2 * u * d(u, 1) + p3 * d(u, 4))

    def estimated_ode(self, cs: jndarray, nudged: jndarray) -> jndarray:
        d = self.d
        p1, p2, p3 = cs
        u = nudged

        return -(p1 * d(u, 2) + p2 * u * d(u, 1) + p3 * d(u, 4))

    def d(self, s: jndarray, m: int) -> jndarray:
        """Compute mth spatial derivative of the state.

        Parameters
        ----------
        s
            System state (e.g., true or nudged) at a point in time
        m
            Number of sptial derivatives to take

        Returns
        -------
        d^n s / d {x^n}
            Approximation of mth spatial derivative of s
        """
        n = len(s)
        k = fft.rfftfreq(n, self._period / n)

        return fft.irfft((1j * k) ** m * fft.rfft(s))
