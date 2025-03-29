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
from base_solver import SinglestepSolver

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
        return self.L(self.gs, true) + self.N(self.gs, true)

    def estimated_ode(self, cs: jndarray, nudged: jndarray) -> jndarray:
        # Note `nudged` should be in frequency domain.
        return self.L(cs, nudged) + self.N(cs, nudged)

        s = nudged
        d = self.d
        f = fft.irfft
        p0, p1, p2 = cs
        return -(p0 * f(d(s, 2)) + p1 * f(s) * f(d(s, 1)) + p2 * f(d(s, 4)))

    def d(self, s: jndarray, m: jndarray) -> jndarray:
        """Compute mth spatial derivative of the state.

        Parameters
        ----------
        s
            System state (e.g., true or nudged) at a point in time in frequency
            domain
        m
            Number of spatial derivatives to take
            If n derivatives are desired, should have shape (n, 1).

        Returns
        -------
        d^m s / d {x^m}
            Approximation of mth spatial derivative of s
        """
        return self.d_coeffs(m) * s

    def d_coeffs(self, m: jndarray) -> jndarray:
        """Compute the coefficient of the mth spatial derivative,
        (2 pi i k)**m.

        Parameters
        ----------
        m
            Number of spatial derivatives to take
            If n derivatives are desired, should have shape (n, 1).
        """
        return (2 * jnp.pi * 1j * self._k) ** m

    @partial(jax.jit, static_argnames="self")
    def _compute_w(self, cs: jndarray, nudged: jndarray) -> jndarray:
        # s = nudged
        # d = self.d
        # m0, m2 = self.L_derivs
        # return (
        #     jnp.stack(
        #         [
        #             -d(s, m0),
        #             -1 / 2 * self.d(fft.rfft(fft.irfft(s) ** 2), 1),
        #             -d(s, m2),
        #         ]
        #     )
        #     .T[self.observed_slice]
        #     .T / self.μ
        # )

        # s = nudged
        # d = self.d
        # f = fft.irfft
        # return (
        #     jnp.stack([-f(d(s, 2)), -f(s) * f(d(s, 1)), -f(d(s, 4))])
        #     .T[self.observed_slice]
        #     .T
        #     / self.μ
        # )

        return (
            jax.jacrev(self.estimated_ode, 0, holomorphic=True)(cs, nudged)[
                self.observed_slice
            ].T
            / self.μ
        )

    def N(self, ps: jndarray, s: jndarray):
        _, p1, _ = ps

        # Alternative computation that runs but the solution looks different
        # return -p1 * fft.rfft(fft.irfft(s) * fft.irfft(self.d(s, 1)))

        return -p1 / 2 * self.d(fft.rfft(fft.irfft(s) ** 2), 1)

    def L(self, ps: jndarray, s: jndarray):
        d = self.d
        p0, _, p2 = ps
        m0, m2 = self.L_derivs

        return -(p0 * d(s, m0) + p2 * d(s, m2))

    L_derivs = jnp.array([2, 4])

    def L_coeffs(self, ps: jndarray) -> jndarray:
        """The coefficients values in

        For each k,
          p0 * d_coeffs(2) + p2 * d_coeffs(2))
        = p0 * (2 pi k)**2 + p2 * (2 pi k)**4.
        """
        ps = jnp.array([ps[0], ps[2]])
        derivs = jnp.expand_dims(self.L_derivs, 1)

        return jnp.sum(ps * self.d_coeffs(derivs).T, axis=1)


class SemiImplicitRK3(SinglestepSolver):
    def __init__(self, system: KSE):
        """

        Reference:
        https://journals.ametsoc.org/view/journals/mwre/134/10/mwr3214.1.xml
        """
        assert isinstance(system, KSE)

        super().__init__(system)

    def _step_factory(self):
        def step(i, vals):
            system = self.system
            L, N, s_ = system.L, system.N, system.observed_slice
            gs = system.gs

            (true, nudged), (dt, cs) = vals
            t = true[i - 1]
            n = nudged[i - 1]

            next_t = jnp.copy(t)
            next_n = jnp.copy(n)
            for k in range(3):
                dtt = dt / (3 - k)

                next_t = t + dtt * N(gs, next_t)
                next_t = (next_t + dtt / 2 * L(gs, t)) / (
                    1 + dtt / 2 * system.L_coeffs(gs)
                )

                next_n = n + dtt * N(cs, next_n)
                next_n = (next_n + dtt / 2 * L(cs, n)) / (
                    1 + dtt / 2 * system.L_coeffs(cs)
                )

            next_n = next_n.at[s_].subtract(
                dt * self.system.μ * (next_n[s_] - next_t[s_])
            )

            true = true.at[i].set(next_t)
            nudged = nudged.at[i].set(next_n)

            return (true, nudged), (dt, cs)

        return step
