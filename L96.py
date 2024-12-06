from jax import numpy as jnp, lax

from base_nudging import System

jndarray = jnp.ndarray


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
