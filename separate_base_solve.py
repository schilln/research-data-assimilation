"""Abstract base classes to simulate `base_system.System`s forward in time."""

from collections.abc import Callable

from jax import numpy as jnp, lax

from separate_base_system import System

jndarray = jnp.ndarray


class Solver:
    def __init__(self, system: System):
        """Base class for solving true and nudged system together."""

        self._system = system
        self._step_true, self._step_nudged = self._step_factory()

    def _step_factory(self) -> Callable:
        """Define step functions to be used in `solve` for the true and nudged
        states.

        Note `step_true` and `step_nudged` will be jitted, so only their
        parameters `i` and `vals` may be updated. Other values (such as
        accessing `system.ode`) will maintain the value used when each function
        is first called.
        """

        def step_true(i, vals):
            raise NotImplementedError()

        def step_nudged(i, vals):
            raise NotImplementedError()

        return step_true, step_nudged

    def _init_solve(
        self,
        state0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        """Initialize the arrays in which to store values computed in `solve`.

        Parameters
        ----------
        state0
            Initial value of system state (true or nudged)
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        true
            Array initialized with inf with the shape to hold N steps of the
            system state
            shape (N, *state0.shape)
        """

        tls = jnp.arange(t0, tf, dt)
        N = len(tls)

        # Store the solution at every step.
        state = jnp.full((N, *state0.shape), jnp.inf)

        # Set initial state.
        state = state.at[0].set(state0)

        return state

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> jndarray:
        """Simulate true state of `self.system` from `t0` to `tf` using step
        size `dt`.

        Note for the following example implementation:
        The values passed to `step` in the `fori_loop` will depend on how `step`
        is defined.

        Example implementation
        ----------------------
        nudged = self._init_solve(true0, t0, tf, dt)

        true, _ = lax.fori_loop(1, len(true), self.step, (true, (dt,)))

        return true

        Parameters
        ----------
        true0
            Initial value of true system state
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        true
            The true states, excluding the initial state(s) `true0`
        """
        raise NotImplementedError()

    def solve_nudged(
        self,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
    ) -> jndarray:
        """Simulate nudged state of `self.system` from `t0` to `tf` using step
        size `dt`.

        Note for the following example implementation:
        The values passed to `step` in the `fori_loop` will depend on how `step`
        is defined.

        Example implementation
        ----------------------
        nudged = self._init_solve(nudged0, t0, tf, dt)

        nudged, _ = lax.fori_loop(
            1,
            len(nudged),
            self.step,
            (nudged, (dt, self.system.cs, true_observed)),
        )

        return nudged

        Parameters
        ----------
        nudged0
            Initial value of nudged system state
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size
        true_observed
            Observed true states

        Returns
        -------
        nudged
            The nudged states, excluding the initial state(s) `nudged0`
        """
        raise NotImplementedError()

    # The following attributes are read-only.
    system = property(lambda self: self._system)
    step_true = property(lambda self: self._step_true)
    step_nudged = property(lambda self: self._step_nudged)


class SinglestepSolver(Solver):
    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> jndarray:
        true = self._init_solve(true0, t0, tf, dt)

        true, _ = lax.fori_loop(1, len(true), self.step_true, (true, (dt,)))

        # Don't return the initial state.
        return true[1:]

    def solve_nudged(
        self,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
    ) -> jndarray:
        nudged = self._init_solve(nudged0, t0, tf, dt)

        nudged, _ = lax.fori_loop(
            1,
            len(nudged),
            self.step_nudged,
            (nudged, (dt, self.system.cs, true_observed)),
        )

        # Don't return the initial state.
        return nudged[1:]


# TODO: This needs to be updated for interp
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

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        start_with_multistep: bool = False,
    ) -> jndarray:
        """

        If `start_with_multistep` is True, then `true0` should have shape
        (k, ...) where k is the number of steps used in the multistep solver,
        and the remaining dimensions are as usual (i.e., contain the state at
        one step).
        """

        # Don't have enough steps to use multistep solver, so use some other
        # solver to start.
        if not start_with_multistep:
            true = self._init_solve(true0, t0, tf, dt)

            # Need k-1 previous steps to use k-step solver.
            # Note upper bound is exclusive, so the span is really
            # [t0, t0 + dt, ..., t0 + dt * (k-1)], for a total of k steps.
            true0 = self._pre_multistep_solver.solve_true(
                true0, t0, t0 + dt * self.k, dt
            )

            true = true.at[1 : self.k].set(true0)

            true, _ = lax.fori_loop(
                self.k, len(true), self.step_true, (true, (dt,))
            )

            # Don't return the initial state.
            return true[1:]
        else:
            true = self._init_solve(true0[0], t0, tf, dt)
            true = true.at[1 : self.k].set(true0[1:])

            true, _ = lax.fori_loop(
                self.k, len(true), self.step_true, (true, (dt,))
            )

            # Don't return the k initial states.
            return true[self.k :]

    def solve_nudged(
        self,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
        start_with_multistep: bool = False,
    ) -> jndarray:
        """

        If `start_with_multistep` is True, then `nudged0` should have shape
        (k, ...) where k is the number of steps used in the multistep solver,
        and the remaining dimensions are as usual (i.e., contain the state at
        one step).
        """

        # Don't have enough steps to use multistep solver, so use some other
        # solver to start.
        if not start_with_multistep:
            nudged = self._init_solve(nudged0, t0, tf, dt)

            # Need k-1 previous steps to use k-step solver.
            # Note upper bound is exclusive, so the span is really
            # [t0, t0 + dt, ..., t0 + dt * (k-1)], for a total of k steps.
            nudged0 = self._pre_multistep_solver.solve_nudged(
                nudged0, t0, t0 + dt * self.k, dt, true_observed
            )

            nudged = nudged.at[1 : self.k].set(nudged)

            nudged, _ = lax.fori_loop(
                self.k,
                len(nudged),
                self.step_nudged,
                (nudged, (dt, self.system.cs, true_observed)),
            )

            # Don't return the initial state.
            return nudged[1:]
        else:
            nudged = self._init_solve(nudged0[0], t0, tf, dt)
            nudged = nudged.at[1 : self.k].set(nudged[1:])

            true, _ = lax.fori_loop(
                self.k,
                len(nudged),
                self.step_nudged,
                (nudged, (dt, self.system.cs, true_observed)),
            )

            # Don't return the k initial states.
            return true[self.k :]

    # The following attribute is read-only.
    k = property(lambda self: self._k)
