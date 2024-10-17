from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

import L96

ndarray = np.ndarray


class RelaxPunch:
    def __init__(
        self,
        I: int,
        J: int,
        J_sim: int,
        Δt: float,
        γ1: float,
        γ2: float,
        ds2: ndarray,
        c1: float,
        c2: float,
        F: float,
        μ: float,
    ):
        """Create a true system and simulated system using Josh Newey's thesis
        (4.9-10). Alternate between evolving the systems and performing
        parameter updates.

        Equation references of the form (JN_#) refer to Josh Newey's thesis and
        those of the form (CGS_#) refer to
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7513031/

        The algorithms for gradient descent come from Josh Newey's thesis.

        When J_sim <= J, the first J_sim values in ds2 will be used in the
        simulated system (J_4.10). J_sim > J is not (yet) supported.

        Parameters
        ----------
        I
            The number of large-scale systems (both true and simulated)
        J
            The number of true small-scale systems
        J_sim
            The number of simulated small-scale systems
        Δt
            The length of time to evolve the true and simulated systems before a
            parameter update.
        γ1
            The true parameter γ1 in Josh Newey's paper (J_4.9)
        γ2
            The true parameter γ2 in Josh Newey's paper (J_4.9)
        ds2
            The J coefficients corresponding to d_j in (J_4.9-10) and
            d_{v_{i, j}} in (CGS_35)
        c1
            The initial value for the parameter c1 in Josh Newey's paper (4.10)
        c2
            The initial value for the parameter c2 in Josh Newey's paper (4.10)
        F
            Forcing term (assume constant)
        μ
            Nudging parameter greater than or equal to zero
        """

        if J_sim > J:
            raise NotImplementedError("J_sim > J is not (yet) supported")

        ds = np.full(I, γ2)
        γs = np.full((I, J), γ1)
        γs2 = np.full(I, γ1)
        ds2 = np.tile(ds2, (I, 1))  # Stack I copies of ds2

        ds_sim = np.full(I, c2)
        γs_sim = np.full((I, J_sim), c1)
        γs2_sim = np.full(I, c1)
        ds2_sim = ds2[:, :J_sim]

        self.system = L96.System(
            I,
            J,
            J_sim,
            ds,
            γs,
            ds2,
            γs2,
            ds_sim,
            γs_sim,
            ds2_sim,
            γs2_sim,
            F,
            μ,
        )

        self.t0, self.Δt = 0, Δt

        self.sols, self.sims = list(), list()

    def _evolve(
        self,
        t0: float,
        Δt: float,
        U0: ndarray,
        V0: ndarray,
        U0_sim: ndarray,
        V0_sim: ndarray,
    ):
        """Evolve the true and simulated system from t0 to t0 + Δt.

        Parameters
        ----------
        t0
            start time
        Δt
            length of time to simulate (i.e., from t0 to t0 + Δt)
        U0
            The state of the true large-scale system at t0
            shape (I,)
        V0
            The states of the true small-scale systems at t0
            shape (I, J)
        U0_sim, V0_sim
            The states of the simulated large- and small-scale systems at t0
            shapes same as those for U0 and V0
        """

        state0 = L96.together(U0, V0)
        state0_sim = L96.together(U0_sim, V0_sim)

        # Evolve true and simulated systems
        sol = solve_ivp(
            self.system.ode_true,
            (t0, t0 + Δt),
            state0,
            dense_output=True,
        )

        sim = solve_ivp(
            self.system.ode_sim,
            (t0, t0 + Δt),
            state0_sim,
            args=(sol.sol,),
            dense_output=True,
        )

        return sol, sim

    def _gradient_descent(
        self,
        t: float,
        sol: Callable[[float], ndarray],
        sim: Callable[[float], ndarray],
        params: ndarray,
        r: float,
    ) -> ndarray:
        """Compute new parameters for the simulated system using gradient
        descent.

        Use (3.13).
        Assume only large-scale states are observed.

        Parameters
        ----------
        t
            The final time of the interval at which point gradient descent is to
            be performed
        sol
            The true solution

            `sol(t)` should have shape (I + I*J,) and be of the form
            [
                u_0, ..., u_{I-1},
                v_{0, 0}, ..., v_{0, J-1},
                v_{1, 0}, ..., v_{1, J-1},
                ...,
                v_{I-1, 0}, ..., v_{I-1, J-1}
            ]
        simsol
            The simulated solution

            `simsol(t)` should have shape (I + I*J,) and be of the form
            [
                u_0, ..., u_{I-1},
                v_{0, 0}, ..., v_{0, J_sim - 1},
                v_{1, 0}, ..., v_{1, J_sim - 1},
                ...,
                v_{I-1, 0}, ..., v_{I-1, J_sim - 1}
            ]
        params
            The unknown parameters to be updated with gradient descent
        r
            The learning rate

        Returns
        -------
        new_params
            The new parameters for the simulated system
        """

        I, J, J_sim, μ = self.I, self.J, self.J_sim, self.μ

        U, _ = L96.apart(sol(t), I, J)
        U_sim, _ = L96.apart(sim(t), I, J_sim)

        new_param0 = params[0] - r * (U_sim - U) @ self._compute_W1(
            t, sim, μ, I, J_sim
        )
        new_param1 = params[1] - r * (U_sim - U) @ self._compute_W2(
            t, sim, μ, I, J_sim
        )

        return np.array([new_param0, new_param1])

    def _compute_W1(
        self,
        t: float,
        simsol: Callable[[float], ndarray],
    ) -> ndarray:
        """Compute w_{k, 1} for k = 0, ..., I-1 using the asymptotic method (4.12).

        Parameters
        ----------
        t
            The time
        simsol
            The nudged/simulated solution.
            `simsol(t)` should have shape (I + I * J_sim,) and be of the form
            [
                u_0, ..., u_{I-1},
                v_{0, 0}, ..., v_{0, J_sim - 1},
                v_{1, 0}, ..., v_{1, J_sim - 1},
                ...,
                v_{I-1, 0}, ..., v_{I-1, J_sim - 1}
            ]

        Returns
        -------
        w_1
            The first sensitivity
            [w_{0, 1}, ..., w_{I-1, 1}]
        """

        U, V = L96.apart(simsol(t), self.I, self.J_sim)

        return (V.sum(axis=1) * U).sum() / self.μ

    def _compute_W2(
        self,
        t: float,
        simsol: Callable[[float], ndarray],
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

        Returns
        -------
        w_2
            The second sensitivity
            [w_{0, 2}, ..., w_{I-1, 2}]
        """

        U, _ = L96.apart(simsol(t), self.I, self.J_sim)

        return -U / self.μ
