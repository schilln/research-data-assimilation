"""Concrete implementations of `base_solver.Solver`."""

from jax import numpy as jnp

from separate_base_system import System
from separate_base_solver import Solver, SinglestepSolver, MultistepSolver

jndarray = jnp.ndarray


class ForwardEuler(SinglestepSolver):
    def __init__(self, system: System):
        """Forward Euler solver.

        See documentation of `base_solver.SinglestepSolver`.
        """
        super().__init__(system)

    def _step_factory(self):
        def step_true(i, vals):
            f = self.system.f_true

            true, (dt,) = vals
            t = true.at[i - 1]

            # TODO: Optimize this.
            t = t.at[:].add(dt * f(t))

            true = true.at[i - 1].set(t)

            return true, (dt,)

        def step_nudged(i, vals):
            f = self.system.f_nudged

            nudged, (dt, cs, true_observed) = vals
            n = nudged.at[i - 1]

            # TODO: Optimize this.
            n = n.at[:].add(dt * f(n, true_observed))

            nudged = nudged.at[i - 1].set(n)

            return nudged, (dt, cs, true_observed)

        return step_true, step_nudged


class TwoStepAdamsBashforth(MultistepSolver):
    def __init__(self, system: System, pre_multistep_solver: Solver):
        """Two-step Adams–Bashforth solver.

        See documentation of `base_solver.MultistepSolver`.

        See https://en.wikipedia.org/wiki/Linear_multistep_method#Two-step_Adams%E2%80%93Bashforth
        """

        super().__init__(system, pre_multistep_solver, 2)

    def _step_factory(self):
        def step_true(i, vals):
            f = self.system.f_true

            true, (dt,) = vals
            t2 = true[i - 2]
            t1 = true[i - 1]

            tmp2 = f(t2)
            tmp1 = f(t1)

            # TODO: Optimize this.
            t1 = t1.at[:].add(3 / 2 * dt * tmp1[0] - 1 / 2 * dt * tmp2[0])

            true = true.at[i].set(t1)

            return true, (dt,)

        def step_nudged(i, vals):
            f = self.system.f_nudged

            nudged, (dt, cs, true_observed) = vals
            t2, n2 = true_observed[i - 2], nudged[i - 2]
            t1, n1 = true_observed[i - 1], nudged[i - 1]

            tmp2 = f(cs, t2, n2)
            tmp1 = f(cs, t1, n1)

            t1 = t1.at[:].add(3 / 2 * dt * tmp1[0] - 1 / 2 * dt * tmp2[0])
            n1 = n1.at[:].add(3 / 2 * dt * tmp1[1] - 1 / 2 * dt * tmp2[1])

            nudged = nudged.at[i].set(n1)

            return nudged, (dt, cs, true_observed)

        return step_true, step_nudged
