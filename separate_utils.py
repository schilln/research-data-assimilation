"""
`run_update` iteratively simulates a `System` and updates parameter values,
returning the sequences of parameter values and nudged-vs-true errors.
"""

from collections.abc import Callable

import numpy as np
from jax import numpy as jnp

import separate_base_system
from separate_base_solver import Solver, SinglestepSolver, MultistepSolver
import base_optim

jndarray = jnp.ndarray


def run_update(
    system: separate_base_system.System,
    true_solver: Solver,
    nudged_solver: Solver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    nudged0: jndarray,
    method: Callable[
        [separate_base_system.System, jndarray, jndarray], jndarray
    ] = base_optim.levenberg_marquardt,
) -> tuple[jndarray, np.ndarray, np.ndarray]:
    """Use `solver` to run `system` and update parameter values with `method`,
    and return sequence of parameter values and errors between nudged and true
    states.

    Parameters
    ----------
    system
        The system to simulate
    true_solver
        An instance of `base_solver.Solver` to simulate true state of `system`
    nudged_solver
        An instance of `base_solver.Solver` to simulate nudged state of `system`
    dt
        The step size to use in `solver`
    T0
        The initial time at which to begin simulation
    Tf
        The (approximate) final time for simulation. This function will use as
        many multiples of `dt` as possible without simulating longer than `Tf`.
    t_relax
        The (approximate) length of time to simulate system between parameter
        updates. This function will use as many multiples of `dt` as possible
        without simulating longer than `t_relax` between parameter updates.
    true0
        The initial state of the true system
    nudged0
        The initial state of the nudged system
    method
        The method to use to perform parameter udpates.

    Returns
    -------
    cs
        The sequence of parameter values
        shape (N + 1, d) where d is the number of parameters being estimated and
            N is the number of parameter updates performed (the first row is the
            initial set of parameter values).
    errors
        The sequence of errors between the true and nudged systems
        shape (N,) where N is the number of parameter updates performed
    tls
        The actual linspace of time values used, in multiples of `t_relax` from
        `T0` to approximately `Tf`
        shape (N + 1,) where N is the number of parameter updates performed
    """
    cs = [system.cs]
    errors = []

    true_args = dict()
    nudged_args = dict()

    if isinstance(true_solver, SinglestepSolver):
        true_k = true_solver.k
        true_args["start_with_multistep"] = True
    else:
        nudged_k = 1

    if isinstance(nudged_solver, MultistepSolver):
        nudged_k = nudged_solver.k
        nudged_args["start_with_multistep"] = True
    else:
        true_k = 1

    t0 = T0
    tf = t0 + t_relax

    true = true_solver.solve_true(true0, t0, tf, dt)
    nudged = nudged_solver.solve_nudged(nudged0, t0, tf, dt)

    true0 = true[-true_k:]
    nudged0 = nudged[-nudged_k:]

    # Update parameters
    system.cs = method(system, true0[system.observed_slice], nudged0)
    cs.append(system.cs)

    t0 = tf
    tf = t0 + t_relax

    # Relative error
    errors.append(np.linalg.norm(true - nudged) / np.linalg.norm(true))

    while tf <= Tf:
        true = true_solver.solve_true(true0, t0, tf, dt, **true_args)
        nudged = nudged_solver.solve_nudged(
            nudged0,
            t0,
            tf,
            dt,
            true[:, system.observed_slice],
            **nudged_args,
        )

        true0 = true[-true_k:]
        nudged0 = nudged[-nudged_k:]

        # Update parameters
        system.cs = method(
            system, true0[-1][system.observed_slice], nudged0[-1]
        )
        cs.append(system.cs)

        t0 = tf
        tf = t0 + t_relax

        # Relative error
        errors.append(np.linalg.norm(true - nudged) / np.linalg.norm(true))
