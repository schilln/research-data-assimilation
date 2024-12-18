"""
Code to run, nudge, and estimate parameters for the two-layer L96 model.
"""

from functools import partial

import jax
from jax import numpy as jnp

from base import System

jndarray = jnp.ndarray


class L96(System):
    def __init__(
        self,
        μ: float,
        bs: jndarray,
        γs: jndarray,
        cs: jndarray,
        observed_slice: slice,
        I: int,
        J: int,
        J_sim: int,
    ):
        super().__init__(μ, bs, γs, cs, observed_slice)

        self._I, self._J, self._J_sim = I, J, J_sim

        # "u slice" and "v slice", for extracting large- and small-scale system
        # states from full state
        self._us, self._vs = jnp.s_[:, 0], jnp.s_[:, 1:]

    def ode(self, true: jndarray) -> jndarray:
        p1, p2 = self.γs
        F, ds = self.bs[0], self.bs[1:]

        u, v = true[self.us], true[self.vs]

        up = (
            jnp.roll(u, 1) * (jnp.roll(u, -1) - jnp.roll(u, 2))
            + p1 * jnp.sum(u * v.T, axis=0)
            - p2 * u
            + F
        )

        vp = -ds * v - jnp.expand_dims(p1 * u**2, axis=1)

        return jnp.concatenate((jnp.expand_dims(up, axis=1), vp), axis=1)

    def estimated_ode(
        self,
        cs: jndarray,
        nudged: jndarray,
    ) -> tuple[jndarray, jndarray]:
        p1, p2 = cs
        F, ds = self.bs[0], self.bs[1:]

        u, v = nudged[self.us], nudged[self.vs]

        up = (
            jnp.roll(u, 1) * (jnp.roll(u, -1) - jnp.roll(u, 2))
            + p1 * jnp.sum(u * v.T, axis=0)
            - p2 * u
            + F
        )

        vp = -ds * v - jnp.expand_dims(p1 * u**2, axis=1)

        return jnp.concatenate((jnp.expand_dims(up, axis=1), vp), axis=1)

    @partial(jax.jit, static_argnames="self")
    def compute_w(self, nudged):
        un, vn = nudged[self.us], nudged[self.vs]

        return jnp.array([jnp.sum(vn, axis=1) * un / self.μ, -un / self.μ])

    # The following attributes are read-only.
    I = property(lambda self: self._I)
    J = property(lambda self: self._J)
    J_sim = property(lambda self: self._J_sim)
    us = property(lambda self: self._us)
    vs = property(lambda self: self._vs)
