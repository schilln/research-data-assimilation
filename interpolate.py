"""
Code to run and nudge the L96 model using jax and jit.
"""

from jax import numpy as jnp, lax
# jax.config.update("jax_enable_x64", True)

# import jax.debug
# from jax.debug import print as jprint

jndarray = jnp.ndarray


class System:
    def __init__(
        self,
        I: int,
        J: int,
        J_sim: int,
        γ1: float,
        γ2: float,
        c1: float,
        c2: float,
        ds: jndarray,
        F: float,
        μ: float,
    ):
        self._I, self._J, self._J_sim = I, J, J_sim

        self._ds = ds
        self._F = F
        self._μ = μ

        self.γ1, self.γ2 = γ1, γ2
        self.c1, self.c2 = c1, c2

    def ode(
        self,
        p1: float,
        p2: float,
        ds: jndarray,
        F: float,
        u: jndarray,
        v: jndarray,
    ) -> tuple[jndarray, jndarray]:
        up = (
            jnp.roll(u, 1) * (jnp.roll(u, -1) - jnp.roll(u, 2))
            + p1 * jnp.sum(u * v.T, axis=0)
            - p2 * u
            + F
        )

        vp = -ds * v - lax.expand_dims(p1 * u**2, (1,))

        return up, vp

    def f_true(self, γ1, γ2, u, v):
        return self.ode(γ1, γ2, self.ds, self.F, u, v)

    def f_nudge(self, c1, c2, u, v, u_true):
        up, vp = self.ode(c1, c2, self.ds, self.F, u, v)
        up -= self.μ * (u - u_true)

        return up, vp

    # These attributes are read-only, while the parameters γ and c may change.
    I = property(lambda self: self._I)
    J = property(lambda self: self._J)
    J_sim = property(lambda self: self._J_sim)
    ds = property(lambda self: self._ds)
    F = property(lambda self: self._F)
    μ = property(lambda self: self._μ)


class RK4:
    def __init__(self, system: System):
        f_true = system.f_true
        f_nudge = system.f_nudge

        def step_true(n, vals):
            """This function will be jitted."""
            (U, V), (dt, γ1, γ2) = vals
            p = γ1, γ2

            u = U[n - 1]
            v = V[n - 1]

            k1u, k1v = f_true(*p, u, v)
            k2u, k2v = f_true(*p, u + dt * k1u / 2, v + dt * k1v / 2)
            k3u, k3v = f_true(*p, u + dt * k2u / 2, v + dt * k2v / 2)
            k4u, k4v = f_true(*p, u + dt * k3u, v + dt * k3v)

            u1 = u + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
            v1 = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

            U = U.at[n].set(u1)
            V = V.at[n].set(v1)

            return (U, V), (dt, γ1, γ2)

        def step_nudge(n, vals):
            """This function will be jitted."""
            (U, V), (dt, c1, c2, U_true), (interp0,) = vals
            p = c1, c2

            u = U[n - 1]
            v = V[n - 1]
            ut0, ut1 = U_true[n - 1], U_true[n]

            k1u, k1v = f_nudge(*p, u, v, ut0)
            k2u, k2v = f_nudge(
                *p, u + dt * k1u / 2, v + dt * k1v / 2, interp0[n - 1]
            )
            k3u, k3v = f_nudge(
                *p, u + dt * k2u / 2, v + dt * k2v / 2, interp0[n - 1]
            )
            k4u, k4v = f_nudge(*p, u + dt * k3u, v + dt * k3v, ut1)

            u1 = u + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
            v1 = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

            U = U.at[n].set(u1)
            V = V.at[n].set(v1)

            return (U, V), (dt, c1, c2, U_true), (interp0,)

        self.step_true = step_true
        self.step_nudge = step_nudge

    def solve(
        self,
        system: System,
        u0: jndarray,
        v0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        U_true: None | jndarray = None,
        interp_method=None,
    ) -> tuple[jndarray, jndarray]:
        tls = jnp.arange(t0, tf, dt)
        N = len(tls)

        # Store the solution at every step.
        I, J = system.I, system.J if U_true is None else system.J_sim
        U = jnp.full((N, I), jnp.inf)
        V = jnp.full((N, I, J), jnp.inf)

        # Set initial state.
        U = U.at[0].set(u0)
        V = V.at[0].set(v0)

        if U_true is None:
            γ1, γ2 = system.γ1, system.γ2
            (U, V), _ = lax.fori_loop(
                1, N, self.step_true, ((U, V), (dt, γ1, γ2))
            )
        else:
            # t + dt/2
            # (Other ODE solvers might use multiple steps, implemented by
            # including additional "interp" values in the last tuple passed to
            # `step_nudge` and using them accordingly within `step_nudge`.)
            interp0 = interp_method(tls, U_true)(tls[:-1] + dt / 2)

            c1, c2 = system.c1, system.c2
            (U, V), _, _ = lax.fori_loop(
                1,
                N,
                self.step_nudge,
                ((U, V), (dt, c1, c2, U_true), (interp0,)),
            )

        return U, V
