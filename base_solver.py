"""Abstract base classes to simulate `base_system.System`s forward in time."""

from collections.abc import Callable

from jax import numpy as jnp, lax

from base_system import System

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


class SinglestepSolver(Solver):
    pass


class MultistepSolver(Solver):
    def __init__(self, system: System, pre_multistep_solver: Solver, k: int):
        """See documentation of `Solver`.

        Parameters
        ----------
        pre_multistep_solver
            An instantiated `Solver` to use until enough steps have been taken
            to use the multistep solver
        k
            The number of steps used in this multistep solver
        """
        super().__init__(system)
        self._k = k

        self._pre_multistep_solver = pre_multistep_solver

    def solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        start_with_multistep: bool = False,
    ) -> tuple[jndarray, jndarray]:
        """

        If `start_with_multistep` is True, then `true0` and `nudged0` should
        have shape (k, ...) where k is the number of steps used in the multistep
        solver, and the remaining dimensions are as usual (i.e., contain the
        state at one step).
        """

        # Don't have enough steps to use multistep solver, so use some other
        # solver to start.
        if not start_with_multistep:
            true, nudged = self._init_solve(true0, nudged0, t0, tf, dt)

            # Need k-1 previous steps to use k-step solver.
            # Note upper bound is exclusive, so the span is really
            # [t0, t0 + dt, ..., t0 + dt * (k-1)], for a total of k steps.
            true0, nudged0 = self._pre_multistep_solver.solve(
                true0, nudged0, t0, t0 + dt * self.k, dt
            )

            true = true.at[1 : self.k].set(true0)
            nudged = nudged.at[1 : self.k].set(nudged0)

            (true, nudged), _ = lax.fori_loop(
                self.k,
                len(true),
                self.step,
                ((true, nudged), (dt, self.system.cs)),
            )

            # Don't return the initial state.
            return true[1:], nudged[1:]
        else:
            true, nudged = self._init_solve(true0[0], nudged0[0], t0, tf, dt)
            true = true.at[1 : self.k].set(true0[1:])
            nudged = nudged.at[1 : self.k].set(nudged0[1:])

            (true, nudged), _ = lax.fori_loop(
                self.k,
                len(true),
                self.step,
                ((true, nudged), (dt, self.system.cs)),
            )

            # Don't return the k initial states.
            return true[self.k :], nudged[self.k :]

    # The following attribute is read-only.
    k = property(lambda self: self._k)
