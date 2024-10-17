from collections.abc import Callable

import numpy as np

ndarray = np.ndarray


class System:
    def __init__(
        self,
        I: int,
        J: int,
        J_sim: int,
        ds: ndarray,
        γs: ndarray,
        ds2: ndarray,
        γs2: ndarray,
        ds_sim: ndarray,
        γs_sim: ndarray,
        ds2_sim: ndarray,
        γs2_sim: ndarray,
        F: float,
        μ: float,
    ):
        """

        Source for Lorenz '96 system:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7513031/
        See equation (35).
        However, I belive this equation has a typo. γ_j should be γ_i.

        Parameters
        ----------
        I
            The number of large-scale systems (both true and simulated)
        J
            The number of true small-scale systems
        J_sim
            The number of simulated small-scale systems
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
        ds_sim
            The coefficients \bar d_i to use in the simulation
            shape: (I,)
        γs_sim
            The coefficients γ_{i, j} to use in the simulation
            shape (I, J)
        ds2_sim
            The coefficients d_{v_{i, j}} to use in the simulation
            shape (I, J)
        γs2_sim
            The coefficients γ_i to use in the simulation
            shape (I,)
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

        self.ds, self.γs, self.ds2, self.γs2 = ds, γs, ds2, γs2

        self.ds_sim, self.γs_sim = ds_sim, γs_sim
        self.ds2_sim, self.γs2_sim = ds2_sim, γs2_sim

        self.F = F
        self.μ = μ

    def __repr__(self):
        return (
            f"I: {self.I} | J: {self.J} | J_sim: {self.J_sim}\n"
            f"F: {self.F} | μ: {self.μ}\n"
            f"ds: {self.ds}\n"
            f"ds2:\n{self.ds2}\n"
            f"γs:\n{self.γs}\n"
            f"γs2: {self.γs2}\n"
            f"ds_sim: {self.ds_sim}\n"
            f"ds2_sim:\n{self.ds2_sim}\n"
            f"γs_sim:\n{self.γs_sim}\n"
            f"γs2_sim: {self.γs2_sim}\n"
        )

    def _ode(
        self,
        state: ndarray,
        J: int,
        ds: ndarray,
        γs: ndarray,
        ds2: ndarray,
        γs2: ndarray,
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

        I = self.I

        # Extract U and V from the state.
        U, V = apart(state, I, J)

        Up, Vp = U, V

        # The time derivatives of the large-scale systems
        Up = self._Uprime(U, V, ds, γs)

        # The time derivatives of the small-scale systems
        Vp = self._Vprime(U, V, ds2, γs2)

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

        return together(
            *self._ode(state, self.J, self.ds, self.γs, self.ds2, self.γs2)
        )

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

        # Extract U and V from the state.
        U, _ = apart(state, self.I, self.J_sim)

        Up, Vp = self._ode(
            state,
            self.J_sim,
            self.ds_sim,
            self.γs_sim,
            self.ds2_sim,
            self.γs2_sim,
        )

        Up -= self._nudge(t, U, state_true)

        return together(Up, Vp)

    def _Uprime(
        self, U: ndarray, V: ndarray, ds: ndarray, γs: ndarray
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
            shape (I, J) where J is either self.J or self.J_sim
            [
                [v_{0, 0}, ..., v_{0, J-1}],
                [v_{1, 0}, ..., v_{1, J-1}],
                ...,
                [v_{I-1, 0}, ..., v_{I-1, J-1}]
            ]
        ds, γs
            See docstring for `__init__`.

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
            + self.F
        )

    def _Vprime(
        self, U: ndarray, V: ndarray, ds2: ndarray, γs2: ndarray
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
            shape (I, J) where J is either self.J or self.J_sim
            [
                [v_{0, 0}, ..., v_{0, J-1}],
                [v_{1, 0}, ..., v_{1, J-1}],
                ...,
                [v_{I-1, 0}, ..., v_{I-1, J-1}]
            ]
        ds2, γs2
            See docstring for `__init__`.

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

        return -ds2 * V - (γs2 * U**2).reshape((self.I, 1))

    def _nudge(
        self,
        t: float,
        U_sim: ndarray,
        state_true: Callable[[float], ndarray],
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
        state_true
            The true state of the system, used to nudge simulated `state`
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
        the nudging term μ * (U_sim - U_true)
        """

        U_true, _ = apart(state_true(t), self.I, self.J)

        return self.μ * (U_sim - U_true)


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
