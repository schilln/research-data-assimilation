"""
`run_update` iteratively simulates a `System` and updates parameter values,
returning the sequences of parameter values and nudged-vs-true errors.
"""

from collections.abc import Callable

import numpy as np
from jax import numpy as jnp

import separate_base_system
from separate_base_solver import Solver, MultistepSolver
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

    if isinstance(true_solver, MultistepSolver):
        true_args["start_with_multistep"] = True
        get_true0 = lambda true: true[-true_solver.k :]
    else:
        get_true0 = lambda true: true[-1]

    if isinstance(nudged_solver, MultistepSolver):
        nudged_args["start_with_multistep"] = True
        get_nudged0 = lambda nudged: nudged[-nudged_solver.k :]
    else:
        get_nudged0 = lambda nudged: nudged[-1]

    t0 = T0
    tf = t0 + t_relax

    true, tls = true_solver.solve_true(true0, t0, tf, dt)
    nudged, _ = nudged_solver.solve_nudged(
        nudged0, t0, tf, dt, true[:, system.observed_slice]
    )

    # Note: If k is -1, the extra first dimension is not eliminated.
    true0 = get_true0(true)
    nudged0 = get_nudged0(nudged)

    # Update parameters
    system.cs = method(system, true[-1][system.observed_slice], nudged[-1])
    cs.append(system.cs)

    t0 = tls[-1]
    tf = t0 + t_relax

    # Relative error
    errors.append(np.linalg.norm(true - nudged) / np.linalg.norm(true))

    while tf <= Tf:
        true, tls = true_solver.solve_true(true0, t0, tf, dt, **true_args)
        nudged, tls = nudged_solver.solve_nudged(
            nudged0,
            t0,
            tf,
            dt,
            true[:, system.observed_slice],
            **nudged_args,
        )

        true0 = get_true0(true)
        nudged0 = get_nudged0(nudged)

        # Update parameters
        system.cs = method(system, true[-1][system.observed_slice], nudged[-1])
        cs.append(system.cs)

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        errors.append(np.linalg.norm(true - nudged) / np.linalg.norm(true))

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return jnp.stack(cs), errors, tls
