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
    def __init__(self, μ: float, bs: jndarray, γs: jndarray, cs: jndarray):
        """
        Parameters
        ----------
        μ
            Nudging parameter
        bs
            Known parameter values of the "true" system, to be used by the
            nudged system as well
        γs
            Unknown parameter values to be used by the "true" system
        cs
            Estimated parameter values to be used by the nudged system (may or
            may not correspond to `γs`)
        """
        self._μ = μ
        self._bs = bs
        self._γs = γs

        self.cs = cs

    def f(
        self,
        cs: jndarray,
        u: jndarray,
        v: jndarray,
        un: jndarray,
        vn: jndarray,
    ) -> tuple[jndarray, jndarray, jndarray, jndarray]:
        """
        Parameters
        ----------
        cs
            Estimated parameter values to be used by the nudged system
        u
            True large-scale systems
        v
            True small-scale systems
        un
            Nudged large-scale systems
        vn
            Nudged small-scale systems

        Returns
        -------
        up, vp, unp, vnp
            The time derivatives of u, v, un, vn
        """
        unp, vnp = self.estimated_ode(cs, un, vn)
        unp -= self.μ * (un - u)

        return *self.ode(u, v), unp, vnp

    def ode(
        self,
        u: jndarray,
        v: jndarray,
    ) -> tuple[jndarray, jndarray]:
        """
        Parameters
        ----------
        u
            True large-scale systems
        v
            True small-scale systems

        Returns
        -------
        up, vp
            The time derivatives of u, v
        """
        raise NotImplementedError()

    def estimated_ode(
        self,
        cs: jndarray,
        u: jndarray,
        v: jndarray,
    ) -> tuple[jndarray, jndarray]:
        """
        Parameters
        ----------
        cs
            Estimated parameter values to be used by the nudged system
        u
            Nudged large-scale systems
        v
            Nudged small-scale systems

        Returns
        -------
        up, vp
            The time derivatives of u, v
        """
        raise NotImplementedError()

    def compute_w(self, un: jndarray, vn: jndarray) -> jndarray:
        """
        Parameters
        ----------
        un, vn
            The nudged large-scale and small-scale systems, respectively

        Returns
        -------
        W
            The ith row corresponds to the asymptotic approximation of the ith
            senstitivity corresponding to the ith unknown parameter ci
        """
        raise NotImplementedError()

    # The following attributes are read-only.
    μ = property(lambda self: self._μ)
    bs = property(lambda self: self._bs)
    γs = property(lambda self: self._γs)


class L96(System):
    def __init__(
        self,
        μ: float,
        bs: jndarray,
        γs: jndarray,
        cs: jndarray,
        I: int,
        J: int,
        J_sim: int,
    ):
        super().__init__(μ, bs, γs, cs)

        self._I, self._J, self._J_sim = I, J, J_sim

    def ode(
        self,
        u: jndarray,
        v: jndarray,
    ) -> tuple[jndarray, jndarray]:
        p1, p2 = self.γs
        F, ds = self.bs[0], self.bs[1:]

        up = (
            jnp.roll(u, 1) * (jnp.roll(u, -1) - jnp.roll(u, 2))
            + p1 * jnp.sum(u * v.T, axis=0)
            - p2 * u
            + F
        )

        vp = -ds * v - lax.expand_dims(p1 * u**2, (1,))

        return up, vp

    def estimated_ode(
        self,
        cs: jndarray,
        u: jndarray,
        v: jndarray,
    ) -> tuple[jndarray, jndarray]:
        p1, p2 = cs
        F, ds = self.bs[0], self.bs[1:]

        up = (
            jnp.roll(u, 1) * (jnp.roll(u, -1) - jnp.roll(u, 2))
            + p1 * jnp.sum(u * v.T, axis=0)
            - p2 * u
            + F
        )

        vp = -ds * v - lax.expand_dims(p1 * u**2, (1,))

        return up, vp

    def compute_w(self, un, vn):
        return jnp.array([jnp.sum(vn, axis=1) * un / self.μ, -un / self.μ])

    # The following attributes are read-only.
    I = property(lambda self: self._I)
    J = property(lambda self: self._J)
    J_sim = property(lambda self: self._J_sim)


def gradient_descent(system: System, u, un, vn, r: float = 1e-4):
    """
    Parameters
    ----------
    r
        Learning rate

    Returns
    -------
    new_cs
        New parameter values cs
    """

    diff = un - u
    gradient = diff @ system.compute_w(un, vn).T

    return system.cs - r * gradient


def levenberg_marquardt(
    system: System, u, un, vn, r: float = 1e-3, λ: float = 1e-2
):
    """
    Parameters
    ----------
    r
        Learning rate
    λ
        Levenberg–Marquardt parameter

    Returns
    -------
    new_cs
        New parameter values cs
    """

    diff = un - u

    gradient = diff @ system.compute_w(un, vn).T
    mat = jnp.outer(gradient, gradient)

    step = jnp.linalg.solve(mat + λ * jnp.eye(len(gradient)), gradient)
    return system.cs - r * step


class RK4:
    def __init__(self, system: System):
        f = system.f

        def step(n, vals):
            """This function will be jitted."""
            (U, V, Un, Vn), (dt, cs) = vals
            u = U[n - 1]
            v = V[n - 1]

            un = Un[n - 1]
            vn = Vn[n - 1]

            k1u, k1v, k1un, k1vn = f(cs, u, v, un, vn)
            k2u, k2v, k2un, k2vn = f(
                cs,
                u + dt * k1u / 2,
                v + dt * k1v / 2,
                un + dt * k1un / 2,
                vn + dt * k1vn / 2,
            )
            k3u, k3v, k3un, k3vn = f(
                cs,
                u + dt * k2u / 2,
                v + dt * k2v / 2,
                un + dt * k2un / 2,
                vn + dt * k2vn / 2,
            )
            k4u, k4v, k4un, k4vn = f(
                cs,
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

            return (U, V, Un, Vn), (dt, cs)

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

        (U, V, Un, Vn), _ = lax.fori_loop(
            1, N, self.step, ((U, V, Un, Vn), (dt, system.cs))
        )

        return U, V, Un, Vn
