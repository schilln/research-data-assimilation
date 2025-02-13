"""
Advection equation u_t + c u_x = 0, x in [0, 2 Pi] with periodic boundary
conditions.

Test case for pseudo-spectral method.

See ACME Volume 4 lab "Spectral 2: A Pseudospectral Method for Periodic
Functions".
"""

from jax import numpy as jnp
from jax.numpy import fft

from base_system import System

jndarray = jnp.ndarray


class Advection(System):
    def ode(self, true: jndarray) -> jndarray:
        return self.gs * Advection.d(true, 1)

    def estimated_ode(self, cs: jndarray, nudged: jndarray) -> jndarray:
        return cs * Advection.d(nudged, 1)

    def d(s: jndarray, m: int) -> jndarray:
        """Compute mth spatial derivative of the state.

        Parameters
        ----------
        s
            System state (e.g., true or nudged) at a point in time
        m
            Number of sptial derivatives to take

        Returns
        -------
        d^m s / d {x^m}
            Approximation of mth spatial derivative of s
        """
        n = len(s)
        k = fft.rfftfreq(n)

        return fft.irfft((1j * k) ** m * fft.rfft(s))
