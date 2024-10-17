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
        """Create a true system and simulated system using Josh Newey's paper
        (4.9-10). Alternate between evolving the systems and performing
        parameter updates.

        Equation references of the form (JN_#) refer to Josh's Newey's paper and
        those of the form (CGS_#) refer to
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7513031/

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
