"""Concrete implementations of `base_solver.Solver`."""

from jax import numpy as jnp

from base_system import System
from base_solver import Solver, SinglestepSolver, MultistepSolver

jndarray = jnp.ndarray


class RK4(SinglestepSolver):
    """4th-order Runge–Kutta solver.

    See documentation of `base_solver.SinglestepSolver`.

    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    def _step_factory(self):
        def step(i, vals):
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


class TwoStepAdamsBashforth(MultistepSolver):
    def __init__(self, system: System, pre_multistep_solver: Solver):
        """Two-step Adams–Bashforth solver.

        See documentation of `base_solver.MultistepSolver`.

        See https://en.wikipedia.org/wiki/Linear_multistep_method#Two-step_Adams%E2%80%93Bashforth
        """
        super().__init__(system, pre_multistep_solver, 2)

    def _step_factory(self):
        def step(i, vals):
            f = self.system.f

            (true, nudged), (dt, cs) = vals
            t2, n2 = true[i - 2], nudged[i - 2]
            t1, n1 = true[i - 1], nudged[i - 1]

            tmp2 = f(cs, t2, n2)
            tmp1 = f(cs, t1, n1)

            t1 = t1.at[:].add(3 / 2 * dt * tmp1[0] - 1 / 2 * dt * tmp2[0])
            n1 = n1.at[:].add(3 / 2 * dt * tmp1[1] - 1 / 2 * dt * tmp2[1])

            true = true.at[i].set(t1)
            nudged = nudged.at[i].set(n1)

            return (true, nudged), (dt, cs)

        return step
