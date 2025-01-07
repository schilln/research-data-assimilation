"""
`run_update` iteratively simulates a `System` and updates parameter values,
returning the sequences of parameter values and nudged-vs-true errors.
"""

from collections.abc import Callable

import numpy as np
from jax import numpy as jnp

import base_system
import simulator

jndarray = jnp.ndarray


def run_update(
    system: base_system.System,
    solver: simulator.Solver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    nudged0: jndarray,
    method: Callable[
        [base_system.System, jndarray, jndarray], jndarray
    ] = base_system.levenberg_marquardt,
) -> tuple[jndarray, np.ndarray, np.ndarray]:
    """Use `solver` to run `system` and update parameter values with `method`,
    and return sequence of parameter values and errors between nudged and true
    states.

    Parameters
    ----------
    system
        The system to simulate
    solver
        An instance of `simulator.Solver` to simulate `system`
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

    if isinstance(solver, simulator.SinglestepSolver):
        return _run_update_singlestep(
            system, solver, dt, T0, Tf, t_relax, true0, nudged0, method
        )
    elif isinstance(solver, simulator.MultistepSolver):
        return _run_update_multistep(
            system, solver, dt, T0, Tf, t_relax, true0, nudged0, method
        )
    else:
        raise NotImplementedError(
            "`solver` should be instance of `simulator.SinglestepSolver` or "
            "`simulator.MultistepSolver`"
        )


def _run_update_singlestep(
    system: base_system.System,
    solver: simulator.SinglestepSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    nudged0: jndarray,
    method: Callable[
        [base_system.System, jndarray, jndarray], jndarray
    ] = base_system.levenberg_marquardt,
) -> tuple[jndarray, np.ndarray, np.ndarray]:
    """Implementation of `run_update` for non-multistep solvers (e.g., RK4),
    here referred to as 'singlestep' solvers. See documentation of `run_update`.
    """

    assert isinstance(solver, simulator.SinglestepSolver)

    cs = [system.cs]
    errors = []

    t0 = T0
    tf = t0 + t_relax
    while tf <= Tf:
        true, nudged = solver.solve(true0, nudged0, t0, tf, dt)

        true0, nudged0 = true[-1], nudged[-1]

        # Update parameters
        system.cs = method(system, true0[system.observed_slice], nudged0)
        cs.append(system.cs)

        t0 = tf
        tf = t0 + t_relax

        # Relative error
        errors.append(np.linalg.norm(true - nudged) / np.linalg.norm(true))

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return jnp.stack(cs), errors, tls


def _run_update_multistep(
    system: base_system.System,
    solver: simulator.MultistepSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    nudged0: jndarray,
    method: Callable[
        [base_system.System, jndarray, jndarray], jndarray
    ] = base_system.levenberg_marquardt,
) -> tuple[jndarray, np.ndarray, np.ndarray]:
    """Implementation of `run_update` for multistep solvers (e.g.,
    Adams–Bashforth). See documentation of `run_update`.
    """

    cs = [system.cs]
    errors = []

    # First iteration
    t0 = T0
    tf = t0 + t_relax

    true, nudged = solver.solve(true0, nudged0, t0, tf, dt)

    true0, nudged0 = true[-solver.k :], nudged[-solver.k :]

    # Update parameters
    system.cs = method(system, true0[-1][system.observed_slice], nudged0[-1])
    cs.append(system.cs)

    t0 = tf
    tf = t0 + t_relax

    # Relative error
    errors.append(np.linalg.norm(true - nudged) / np.linalg.norm(true))

    while tf <= Tf:
        true, nudged = solver.solve(
            true0,
            nudged0,
            t0,
            tf,
            dt,
            start_with_multistep=True,
        )

        true0, nudged0 = true[-solver.k :], nudged[-solver.k :]

        # Update parameters
        system.cs = method(
            system, true0[-1][system.observed_slice], nudged0[-1]
        )
        cs.append(system.cs)

        t0 = tf
        tf = t0 + t_relax

        # Relative error
        errors.append(np.linalg.norm(true - nudged) / np.linalg.norm(true))

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return jnp.stack(cs), errors, tls
