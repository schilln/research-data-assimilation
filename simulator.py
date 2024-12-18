"""
Classes to simulate `System`s forward in time.
"""

from collections.abc import Callable

from jax import numpy as jnp, lax

from base import System

jndarray = jnp.ndarray


class Solver:
    def __init__(self, system: System):
        """Base class for solving true and nudged system together."""

        self._system = system
        self._step = self._step_factory()

    def _step_factory(self) -> Callable:
        """Define a step function to be used in `solve`.

        Note `step` will be jitted, so only its parameters `i` and `vals` may be
        updated. Other values (such as accessing `system.ode`) will maintain
        the value used when `step` is first called.
        """

        def step(i, vals):
            """This function will be jitted."""
            raise NotImplementedError()

        return step

    def _init_solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        """Initialize the arrays in which to store values computed in `solve`.

        Parameters
        ----------
        true0
            Initial value of true system state
        nudged0
            Initial value of nudged system state
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size
        """

        tls = jnp.arange(t0, tf, dt)
        N = len(tls)

        # Store the solution at every step.
        true = jnp.full((N, *true0.shape), jnp.inf)
        nudged = jnp.full((N, *nudged0.shape), jnp.inf)

        # Set initial state.
        true = true.at[0].set(true0)
        nudged = nudged.at[0].set(nudged0)

        return true, nudged

    def solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        """Simulate `self.system` from `t0` to `tf` using step size `dt`.

        Note for the following example implementation:
        The values passed to `step` in the `fori_loop` will depend on how `step`
        is defined.

        Example implementation
        ----------------------
        true, nudged = self._init_solve(true0, nudged0, t0, tf, dt)

        (true, nudged), _ = lax.fori_loop(
            1, len(true), self.step, ((true, nudged), (dt, self.system.cs))
        )

        return true, nudged

        Parameters
        ----------
        true0
            Initial value of true system state
        nudged0
            Initial value of nudged system state
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size
        """
        raise NotImplementedError()

    # The following attributes are read-only.
    system = property(lambda self: self._system)
    step = property(lambda self: self._step)


class RK4(Solver):
    def __init__(self, system: System):
        super().__init__(system)

    def solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        true, nudged = self._init_solve(true0, nudged0, t0, tf, dt)

        (true, nudged), _ = lax.fori_loop(
            1, len(true), self.step, ((true, nudged), (dt, self.system.cs))
        )

        return true, nudged

    def _step_factory(self):
        def step(i, vals):
            """This function will be jitted."""

            f = self.system.f

            (true, nudged), (dt, cs) = vals
            t = true[i - 1]
            n = nudged[i - 1]

            k1t, k1n = f(cs, t, n)
            k2t, k2n = f(
                cs,
                t + dt * k1t / 2,
                n + dt * k1n / 2,
            )
            k3t, k3n = f(
                cs,
                t + dt * k2t / 2,
                n + dt * k2n / 2,
            )
            k4t, k4n = f(
                cs,
                t + dt * k3t,
                n + dt * k3n,
            )

            t = t.at[:].add((dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t))
            n = n.at[:].add((dt / 6) * (k1n + 2 * k2n + 2 * k3n + k4n))

            true = true.at[i].set(t)
            nudged = nudged.at[i].set(n)

            return (true, nudged), (dt, cs)

        return step
