"""
The specialized L96 model used in Josh Newey's thesis.
Specifically, certain parameters are restricted to be equal.
"""

from collections.abc import Callable

import numpy as np

ndarray = np.ndarray


class System:
    def __init__(
        self,
        I: int,
        J: int,
        J_sim: int,
        γ1: float,
        γ2: float,
        c1: float,
        c2: float,
        ds: ndarray,
        F: float,
        μ: float,
    ):
        """

        Equation references are to Josh Newey's thesis.

        Parameters
        ----------
        I
            The number of large-scale systems (both true and simulated)
        J
            The number of true small-scale systems
        J_sim
            The number of simulated small-scale systems
        γ1
            The true parameter γ_1 in (4.9)
        γ2
            The true parameter γ_2 in (4.9)
        c1
            The initial value for the parameter c_1 (simulated version of γ1) in
            (4.9)
        c2
            The initial value for the parameter c_2 (simulated version of γ2) in
            (4.9)
        ds
            The coefficients d_j in (4.9–10)
            shape (J,)
        F
            Forcing term (assume constant)
        μ
            Nudging parameter greater than or equal to zero

        Attributes
        ----------
        All of the parameters to `__init__`

        Methods
        -------
        ode_true
            Compute the time derivative of the true system.
        ode_sim
            Compute the time derivative of the simulated system.
        """

        self.I, self.J, self.J_sim = I, J, J_sim

        self.γ1, self.γ2 = γ1, γ2
        self.c1, self.c2 = c1, c2
        self.ds = ds

        self.F = F
        self.μ = μ

    def __repr__(self):
        return (
            f"I: {self.I} | J: {self.J} | J_sim: {self.J_sim}\n"
            f"F: {self.F} | μ: {self.μ}\n"
            f"γ1:\n{self.γ1}\n"
            f"γ2: {self.γ2}\n"
            f"c1:\n{self.c1}\n"
            f"c2: {self.c2}\n"
            f"ds: {self.ds}\n"
        )

    def _ode(
        self,
        state: ndarray,
        J: int,
    ) -> tuple[ndarray, ndarray]:
        """Compute u'_i and v'_{i, j} for each i and j.

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
        J
            The number of small-scale systems

        Returns
        -------
        U'
            the derivative dU/dt
            shape (I,)
            [u'_0, ..., u'_{I-1}]
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

        # Extract U and V from the state.
        U, V = apart(state, self.I, J)

        # The time derivatives of the large-scale systems
        Up = self._Uprime(U, V)

        # The time derivatives of the small-scale systems
        Vp = self._Vprime(U, V)

        return Up, Vp

    def ode_true(
        self,
        t: float,
        state: ndarray,
    ) -> ndarray:
        """Compute u'_i and v'_{i, j} for each i and j.

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

        return together(*self._ode(state, self.J))

    def ode_sim(
        self,
        t: float,
        state: ndarray,
        state_true: Callable[[float], ndarray],
    ) -> ndarray:
        """Compute u'_i and v'_{i, j} for each i and j, using μ to nudge the
        simulated state.

        Parameters
        ----------
        t
            The time
        state
            The concatenation of large-scale and small-scale systems
            shape (I + I * J_sim,)
            [
                u_0, ..., u_{I-1},
                v_{0, 0}, ..., v_{0, J_sim - 1},
                v_{1, 0}, ..., v_{1, J_sim - 1},
                ...,
                v_{I-1, 0}, ..., v_{I-1, J_sim - 1}
            ]
        state_true
            The true state of the system, used to nudge simulated `state`.
            `state_true(t)` should have shape (I + I*J,)
            [
                u_0, ..., u_{I-1},
                v_{0, 0}, ..., v_{0, J-1},
                v_{1, 0}, ..., v_{1, J-1},
                ...,
                v_{I-1, 0}, ..., v_{I-1, J-1}
            ]

        Returns
        -------
        the time derivative of state
            shape (I + I * J_sim,)
            [
                u'_0, ..., u'_{I-1},
                v'_{0, 0}, ..., v'_{0, J_sim - 1},
                v'_{1, 0}, ..., v'_{1, J_sim - 1},
                ...,
                v'_{I-1, 0}, ..., v'_{I-1, J_sim - 1}
            ]
        """

        Up, Vp = self._ode(
            state,
            self.J_sim,
        )

        # Extract U_sim from the simulated state and U_true from the true state.
        U_sim = state[: self.I]
        U_true = state_true(t)[: self.I]

        Up -= self.μ * (U_sim - U_true)

        return together(Up, Vp)

    def _Uprime(self, U: ndarray, V: ndarray) -> ndarray:
        """Return the time derivatives of the large-scale systems.

        Parameters
        ----------
        U
            The concatenation of u terms
            shape (I,)
            [u_0, ..., u_{I-1}]
        V
            The array of v terms
            shape (I, J) where J is either self.J or self.J_sim
            [
                [v_{0, 0}, ..., v_{0, J-1}],
                [v_{1, 0}, ..., v_{1, J-1}],
                ...,
                [v_{I-1, 0}, ..., v_{I-1, J-1}]
            ]

        Returns
        -------
        U'
            the derivative dU/dt
            shape (I,)
            [u'_0, ..., u'_{I-1}]
        """

        return (
            # The following line uses the correct version of the two-layer L96
            # system (Josh Newey's thesis has a typo).
            np.roll(U, 1) * (np.roll(U, -1) - np.roll(U, 2))
            + self.γ1 * (U * V.T).sum(axis=0)
            - self.γ2 * U
            + self.F
        )

    def _Vprime(self, U: ndarray, V: ndarray) -> ndarray:
        """Return the time derivatives of the small-scale systems.

        Parameters
        ----------
        U
            The concatenation of u terms
            shape (I,)
            [u_0, ..., u_{I-1}]
        V
            The array of v terms
            shape (I, J) where J is either self.J or self.J_sim
            [
                [v_{0, 0}, ..., v_{0, J-1}],
                [v_{1, 0}, ..., v_{1, J-1}],
                ...,
                [v_{I-1, 0}, ..., v_{I-1, J-1}]
            ]

        Returns
        -------
        V'
            the derivative dV/dt
            shape (I, J) where J is either self.J or self.J_sim
            [
                [v'_{0, 0}, ..., v'_{0, J-1}],
                [v'_{1, 0}, ..., v'_{1, J-1}],
                ...,
                [v'_{I-1, 0}, ..., v'_{I-1, J-1}]
            ]
        """

        return -self.ds * V - (self.γ1 * U**2).reshape((self.I, 1))


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
