"""
Advection equation u_t + c u_x = 0, x in [0, 2 Pi] with periodic boundary
conditions.

Test case for pseudospectral method.

See ACME Volume 4 lab "Spectral 2: A Pseudospectral Method for Periodic
Functions".
"""

from jax import numpy as jnp
from jax.numpy import fft

from base_system import System

jndarray = jnp.ndarray


class Advection(System):
    def ode(self, true: jndarray) -> jndarray:
        u = true
        n = len(u)
        k = fft.rfftfreq(n)

        return self.gs * fft.irfft(1j * k * fft.rfft(u))

    def estimated_ode(self, cs: jndarray, nudged: jndarray) -> jndarray:
        v = nudged
        n = len(v)
        k = fft.rfftfreq(n)

        return cs * fft.irfft(1j * k * fft.rfft(v))
