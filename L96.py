from collections.abc import Callable

import numpy as np

ndarray = np.ndarray


# Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7513031/
# See equation (35).


def ode(
    t: float,
    state: ndarray,
    I: int,
    J: int,
    ds: ndarray,
    γs: ndarray,
    ds2: ndarray,
    γs2: ndarray,
    F: float,
    μ: float | None = None,
    state_true: Callable[[float], ndarray] | None = None,
    J_true: int | None = None,
) -> tuple[ndarray, ndarray]:
    """Compute u'_i and v'_{i, j} for each i and j.

    If μ is not None, expect `state_true` and use μ to nudge the simulated
    `state`.

    Parameters
    ----------
    t
        The time
    state
        The concatenation of large-scale and small-scale systems
        shape (I + I*J,)
        [
            u_0, ..., u_{I-1},
            v_{0, 0}, ..., v_{0, J-1},
            v_{1, 0}, ..., v_{1, J-1},
            ...,
            v_{I-1, 0}, ..., v_{I-1, J-1}
        ]
    I
        The number of large-scale systems
    J
        The number of small-scale systems
    ds
        The coefficients \bar d_i
        shape: (I,)
    γs
        The coefficients γ_{i, j}
        shape (I, J)
    ds2
        The coefficients d_{v_{i, j}}
        shape (I, J)
    γs2
        The coefficients γ_i
        shape (I,)
        (I think equation (35) has a typo. γ_j should be γ_i.)
    F
        forcing term (assume constant)
    μ
        Nudging parameter greater than or equal to zero
    state_true
        The true state of the system, used to nudge simulated `state`.
        Should be callable with time t, returning the same shape and format as
        `state` except that J_true (the number of true small-scale systems) may
        differ from J (the number of simulated small-scale systems).
    J_true
        The number of true small-scale systems

    Returns
    -------
    the time derivative of state
        shape (I + I*J,)
        [
            u'_0, ..., u'_{I-1},
            v'_{0, 0}, ..., v'_{0, J-1},
            v'_{1, 0}, ..., v'_{1, J-1},
            ...,
            v'_{I-1, 0}, ..., v'_{I-1, J-1}
        ]
    """

    # Extract U and V from the state.
    U, V = apart(state, I, J)

    Up, Vp = U, V

    # The time derivatives of the large-scale systems
    Up = Uprime(U, V, ds, γs, F)

    # The time derivatives of the small-scale systems
    Vp = Vprime(U, V, ds2, γs2)

    if μ is not None:
        Up -= nudge(t, U, μ, state_true, J_true)

    return together(Up, Vp)


def Uprime(
    U: ndarray, V: ndarray, ds: ndarray, γs: ndarray, F: float
) -> ndarray:
    """Return the time derivatives of the large-scale systems.

    Parameters
    ----------
    U
        The concatenation of u terms
        shape (I,)
        [u_0, ..., u_{I-1}]
    V
        The array of v terms
        shape (I, J)
        [
            [v_{0, 0}, ..., v_{0, J-1}],
            [v_{1, 0}, ..., v_{1, J-1}],
            ...,
            [v_{I-1, 0}, ..., v_{I-1, J-1}]
        ]
    ds
        The coefficients \bar d_i
        shape: (I,)
    γs
        The coefficients γ_{i, j}
        shape (I, J)
    F
        forcing term (assume constant)

    Returns
    -------
    U'
        the derivative dU/dt
        shape (I,)
        [u'_0, ..., u'_{I-1}]
    """

    return (
        np.roll(U, 1) * (np.roll(U, -1) - np.roll(U, 2))
        + (U * (γs * V).T).sum(axis=0)
        - ds * U
        + F
    )


def Vprime(
    U: ndarray, V: ndarray, ds2: ndarray, γs2: ndarray, I: int
) -> ndarray:
    """Return the time derivatives of the small-scale systems.

    Parameters
    ----------
    U
        The concatenation of u terms
        shape (I,)
        [u_0, ..., u_{I-1}]
    V
        The array of v terms
        shape (I, J)
        [
            [v_{0, 0}, ..., v_{0, J-1}],
            [v_{1, 0}, ..., v_{1, J-1}],
            ...,
            [v_{I-1, 0}, ..., v_{I-1, J-1}]
        ]
    ds2
        The coefficients d_{v_{i, j}}
        shape (I, J)
    γs2
        The coefficients γ_j
        shape (I,)
    I
        The number of large-scale systems

    Returns
    -------
    V'
        the derivative dV/dt
        shape (I, J)
        [
            [v'_{0, 0}, ..., v'_{0, J-1}],
            [v'_{1, 0}, ..., v'_{1, J-1}],
            ...,
            [v'_{I-1, 0}, ..., v'_{I-1, J-1}]
        ]
    """

    return -ds2 * V - (γs2 * U**2).reshape((I, 1))


def nudge(
    t: float,
    U_sim: ndarray,
    μ: float,
    state_true: Callable[[float], ndarray],
    I: int,
    J_true: int,
) -> ndarray:
    """Return the nudging term μ * (U_sim - U_true).

    Parameters
    ----------
    t
        The time
    U_sim
        The simulated large-scale system state
        shape (I,)
        [u_0, ..., u_{I-1}]
    μ
        Nudging parameter greater than or equal to zero
    state_true
        The true state of the system, used to nudge simulated `state`.
        Should be callable with time t. See docstring of `ode` for shape.
    I
        The number of large-scale systems
    J_true
        The number of true small-scale systems

    Returns
    -------
    the nudging term μ * (U_sim - U_true)
    """

    if state_true is None or J_true is None:
        raise ValueError(
            "`state_true` and `J_true` should not be None if μ is not None"
        )

    U_true, _ = apart(state_true(t), I, J_true)

    return μ * (U_sim - U_true)


def together(U: ndarray, V: ndarray) -> ndarray:
    """Concatenate all u and v terms.

    Parameters
    ----------
    U
        The concatenation of u terms
        shape (I,)
        [u_0, ..., u_{I-1}]
    V
        The array of v terms
        shape (I, J)
        [
            [v_{0, 0}, ..., v_{0, J-1}],
            [v_{1, 0}, ..., v_{1, J-1}],
            ...,
            [v_{I-1, 0}, ..., v_{I-1, J-1}]
        ]

    Returns
    -------
    state
        The concatenation of large-scale and small-scale systems
        shape (I + I*J,)
        [
            u_0, ..., u_{I-1},
            v_{0, 0}, ..., v_{0, J-1},
            v_{1, 0}, ..., v_{1, J-1},
            ...,
            v_{I-1, 0}, ..., v_{I-1, J-1}
        ]
    """

    return np.concatenate((U, V.ravel()))


def apart(state: ndarray, I: int, J: int) -> tuple[ndarray, ndarray]:
    """Extract U and V from state.

    Parameters
    ----------
    state
        The concatenation of large-scale and small-scale systems
        shape (I + I*J,)
        [
            u_0, ..., u_{I-1},
            v_{0, 0}, ..., v_{0, J-1},
            v_{1, 0}, ..., v_{1, J-1},
            ...,
            v_{I-1, 0}, ..., v_{I-1, J-1}
        ]
    I
        The number of large-scale systems
    J
        The number of small-scale systems

    Returns
    -------
    U
        The concatenation of u terms
        shape (I,)
        [u_0, ..., u_{I-1}]
    V
        The array of v terms
        shape (I, J)
        [
            [v_{0, 0}, ..., v_{0, J-1}],
            [v_{1, 0}, ..., v_{1, J-1}],
            ...,
            [v_{I-1, 0}, ..., v_{I-1, J-1}]
        ]
    """

    # Extract the large-scale systems.
    U = state[:I]

    # Extract the small-scale systems. The ith row contains the J small-scale
    # systems for the ith large-scale system u_i.
    V = state[I:].reshape((I, J))

    return U, V


def compute_W1(
    t: float, simsol: Callable[[float], ndarray], μ: float, I: int, J_sim: int
) -> ndarray:
    """Compute w_{k, 1} for k = 0, ..., I-1 using the asymptotic method (4.12).

    Parameters
    ----------
    t
        The time
    simsol
        The nudged/simulated solution.
        `simsol(t)` should have shape (I + I*J_sim,) and be of the form
        [
            u_0, ..., u_{I-1},
            v_{0, 0}, ..., v_{0, J_sim-1},
            v_{1, 0}, ..., v_{1, J_sim-1},
            ...,
            v_{I-1, 0}, ..., v_{I-1, J_sim-1}
        ]
    μ
        The nudging parameter
    I
        The number of large-scale systems
    J_sim
        The number of small-scale systems

    Returns
    -------
    w_1
        The first sensitivity
        [w_{0, 1}, ..., w_{I-1, 1}]
    """

    U, V = apart(simsol(t), I, J_sim)

    return (V.sum(axis=1) * U).sum() / μ


def compute_W2(
    t: float, simsol: Callable[[float], ndarray], μ: float, I: int, J_sim: int
) -> ndarray:
    """Compute w_{k, 2} for k = 0, ..., I-1 using the asymptotic method (4.12).

    Parameters
    ----------
    t
        The time
    simsol
        The nudged/simulated solution.
        `simsol(t)` should have shape (I + I*J_sim,) and be of the form
        [
            u_0, ..., u_{I-1},
            v_{0, 0}, ..., v_{0, J_sim-1},
            v_{1, 0}, ..., v_{1, J_sim-1},
            ...,
            v_{I-1, 0}, ..., v_{I-1, J_sim-1}
        ]
    μ
        The nudging parameter
    I
        The number of large-scale systems
    J_sim
        The number of small-scale systems

    Returns
    -------
    w_2
        The second sensitivity
        [w_{0, 2}, ..., w_{I-1, 2}]
    """

    U, _ = apart(simsol(t), I, J_sim)

    return -U / μ


def gradient_descent(
    t: float,
    sol: Callable[[float], ndarray],
    sim: Callable[[float], ndarray],
    params: ndarray,
    r: float,
    μ: float,
    I: int,
    J: int,
    J_sim: int,
) -> ndarray:
    """Compute new parameters for the simulated system using gradient descent.

    Use (3.13).
    Assume only large-scale states are observed.

    Parameters
    ----------
    t
        The final time of the interval at which point gradient descent is to be
        performed
    sol
        The true solution

        `sol(t)` should have shape (I + I*J,) and be of the form
        [
            u_0, ..., u_{I-1},
            v_{0, 0}, ..., v_{0, J_sim-1},
            v_{1, 0}, ..., v_{1, J_sim-1},
            ...,
            v_{I-1, 0}, ..., v_{I-1, J_sim-1}
        ]
    simsol
        The simulated solution

        `simsol(t)` should have the same shape as `sol(t)` except the the
        dimension `J` should be `J_sim` here.
    params
        The unknown parameters to be updated with gradient descent
    r
        The learning rate
    μ
        The nudging parameter
    I
        The number of large-scale systems (both true and simulated)
    J
        The number of true small-scale systems
    J_sim
        The number of simulated small-scale systems

    Returns
    -------
    new_params
        The new parameters for the simulated system
    """

    U, _ = apart(sol(t), I, J)
    U_sim, _ = apart(sim(t), I, J_sim)

    new_param0 = params[0] - r * (U_sim - U) @ compute_W1(t, sim, μ, I, J_sim)
    new_param1 = params[1] - r * (U_sim - U) @ compute_W2(t, sim, μ, I, J_sim)

    return np.array([new_param0, new_param1])
