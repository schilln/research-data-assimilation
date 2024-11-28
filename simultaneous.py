"""
Code to run and nudge the L96 model using jax and jit, with the true and nudged
systems solved together rather than separately.
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

    def f(self, γ1, γ2, c1, c2, u, v, un, vn):
        """
        Parameters
        ----------
        un
            Nudged large-scale systems
        vn
            Nudged small-scale systems
        """
        unp, vnp = self.ode(c1, c2, self.ds, self.F, un, vn)
        unp -= self.μ * (un - u)

        return *self.ode(γ1, γ2, self.ds, self.F, u, v), unp, vnp

    def compute_w1(self, u, v):
        """
        Parameters
        ----------
        U, V
            The nudged large-scale and small-scale systems
        """

        return jnp.sum(v, axis=1) * u / self.μ

    def compute_w2(self, u):
        """
        Parameters
        ----------
        U
            The nudged large-scale system
        """

        return -u / self.μ

    # These attributes are read-only, while the parameters γ and c may change.
    I = property(lambda self: self._I)
    J = property(lambda self: self._J)
    J_sim = property(lambda self: self._J_sim)
    ds = property(lambda self: self._ds)
    F = property(lambda self: self._F)
    μ = property(lambda self: self._μ)


def gradient_descent(system: System, u, un, vn, r: float):
    """
    Parameters
    ----------
    r
        Learning rate
    """

    diff = un - u
    gradient = jnp.array(
        [diff @ system.compute_w1(un, vn), diff @ system.compute_w2(un)]
    )

    return system.c1 - r * gradient[0], system.c2 - r * gradient[1]


class RK4:
    def __init__(self, system: System):
        f = system.f

        def step(n, vals):
            """This function will be jitted."""
            (U, V, Un, Vn), (dt, γ1, γ2, c1, c2) = vals
            p = γ1, γ2
            q = c1, c2

            u = U[n - 1]
            v = V[n - 1]

            un = Un[n - 1]
            vn = Vn[n - 1]

            k1u, k1v, k1un, k1vn = f(*p, *q, u, v, un, vn)
            k2u, k2v, k2un, k2vn = f(
                *p,
                *q,
                u + dt * k1u / 2,
                v + dt * k1v / 2,
                un + dt * k1un / 2,
                vn + dt * k1vn / 2,
            )
            k3u, k3v, k3un, k3vn = f(
                *p,
                *q,
                u + dt * k2u / 2,
                v + dt * k2v / 2,
                un + dt * k2un / 2,
                vn + dt * k2vn / 2,
            )
            k4u, k4v, k4un, k4vn = f(
                *p,
                *q,
                u + dt * k3u,
                v + dt * k3v,
                un + dt * k3un,
                vn + dt * k3vn,
            )

            u1 = u + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
            v1 = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

            u1n = un + (dt / 6) * (k1un + 2 * k2un + 2 * k3un + k4un)
            v1n = vn + (dt / 6) * (k1vn + 2 * k2vn + 2 * k3vn + k4vn)

            U = U.at[n].set(u1)
            V = V.at[n].set(v1)

            Un = Un.at[n].set(u1n)
            Vn = Vn.at[n].set(v1n)

            return (U, V, Un, Vn), (dt, γ1, γ2, c1, c2)

        self.step = step

    def solve(
        self,
        system: System,
        u0: jndarray,
        v0: jndarray,
        un0: jndarray,
        vn0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray, jndarray]:
        tls = jnp.arange(t0, tf, dt)
        N = len(tls)

        # Store the solution at every step.
        I, J, J_sim = system.I, system.J, system.J_sim
        U = jnp.full((N, I), jnp.inf)
        V = jnp.full((N, I, J), jnp.inf)
        Un = jnp.full((N, I), jnp.inf)
        Vn = jnp.full((N, I, J_sim), jnp.inf)

        # Set initial state.
        U = U.at[0].set(u0)
        V = V.at[0].set(v0)
        Un = U.at[0].set(un0)
        Vn = V.at[0].set(vn0)

        γ1, γ2 = system.γ1, system.γ2
        c1, c2 = system.c1, system.c2
        (U, V, Un, Vn), _ = lax.fori_loop(
            1, N, self.step, ((U, V, Un, Vn), (dt, γ1, γ2, c1, c2))
        )

        return U, V, Un, Vn
