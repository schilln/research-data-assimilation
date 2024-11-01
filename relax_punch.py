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
        γ1: float,
        γ2: float,
        c1: float,
        c2: float,
        ds: ndarray,
        F: float,
        μ: float,
    ):
        """Create a true system and simulated system using Josh Newey's thesis
        (4.9-10). Alternate between evolving the systems and performing
        parameter updates.

        The algorithms for gradient descent come from Josh Newey's thesis.

        When J_sim <= J, the first J_sim values in ds will be used in the
        simulated system (4.10). J_sim > J is not (yet) supported.

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
        """

        if J_sim > J:
            raise NotImplementedError("J_sim > J is not (yet) supported")

        self.system = L96.System(I, J, J_sim, γ1, γ2, c1, c2, ds, F, μ)

    def iterate(
        self,
        Δt: float,
        num_iters: int,
        learning_rate: float,
        U0: ndarray,
        V0: ndarray,
        U0_sim: ndarray,
        V0_sim: ndarray,
        options: dict = {},
        param_update_method: Callable | None = None,
    ):
        """

        Parameters
        ----------
        Δt
            The length of time to evolve the true and simulated systems before a
            parameter update
        num_iters
            The number of iterations of evolving the system and updating
            parameters to perform
        learning_rate
            The learning rate to use when updating parameters
        U0
            The state of the true large-scale system at t0
            shape (I,)
        V0
            The states of the true small-scale systems at t0
            shape (I, J)
        U0_sim, V0_sim
            The states of the simulated large- and small-scale systems at t0
            shapes same as those for U0 and V0
        options
            Additional keyword arguments to be passed to `solve_ivp`
        param_update_method
            One of `RelaxPunch`'s parameter update methods;
            for example, `rp._gradient_descent`

        Saves
        -----
        sols
            A list of `solve_ivp` solutions to the true solution
            length: `num_iters`
        sims
            A list of `solve_ivp` solutions to the simulated solution
            length: `num_iters`
        sol
            A callable piecing together true solutions from all time intervals
        sim
            A callable piecing together simulated solutions from all time
            intervals
        c1s
            An ndarray of the consecutive values found for `c1`
        c2s
            An ndarray of the consecutive values found for `c2`
        """

        # Default to gradient descent
        if param_update_method is None:
            param_update_method = self._gradient_descent

        system = self.system

        self.sols, self.sims = list(), list()

        self.c1s = np.full(num_iters + 1, np.inf)
        self.c2s = np.full(num_iters + 1, np.inf)

        self.c1s[0], self.c2s[0] = system.c1, system.c2

        t0 = 0
        for i in range(num_iters):
            sol, sim = self.evolve(
                t0,
                Δt,
                U0,
                V0,
                U0_sim,
                V0_sim,
                options,
            )

            self.sols.append(sol)
            self.sims.append(sim)

            c1, c2 = param_update_method(
                t0 + Δt, sol.sol, sim.sol, system.c1, system.c2, learning_rate
            )

            self.c1s[i + 1] = c1
            self.c2s[i + 1] = c2
            self._update_params(c1, c2)

            t0 += Δt
            U0, V0 = L96.apart(sol.y.T[-1], system.I, system.J)
            U0_sim, V0_sim = L96.apart(sim.y.T[-1], system.I, system.J_sim)

        self._concatenate_sols(Δt, num_iters)

    def evolve(
        self,
        t0: float,
        Δt: float,
        U0: ndarray,
        V0: ndarray,
        U0_sim: ndarray,
        V0_sim: ndarray,
        options: dict = {},
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
        options
            Additional keyword arguments to be passed to `solve_ivp`
        """

        state0 = L96.together(U0, V0)
        state0_sim = L96.together(U0_sim, V0_sim)

        # Evolve true and simulated systems
        sol = solve_ivp(
            self.system.ode_true,
            (t0, t0 + Δt),
            state0,
            dense_output=True,
            **options,
        )

        sim = solve_ivp(
            self.system.ode_sim,
            (t0, t0 + Δt),
            state0_sim,
            args=(sol.sol,),
            dense_output=True,
            **options,
        )

        return sol, sim

    def _update_params(self, c1, c2):
        self.system.c1 = c1
        self.system.c2 = c2

    def _gradient_descent(
        self,
        t: float,
        sol: Callable[[float], ndarray],
        sim: Callable[[float], ndarray],
        c1: float,
        c2: float,
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
        c1
            The parameter c1 to update
        c2
            The parameter c2 to update
        r
            The learning rate

        Returns
        -------
        new_params
            The new parameters for the simulated system
        """

        system = self.system

        U = sol(t)[: system.I]
        U_sim = sim(t)[: system.I]

        new_c1 = c1 - r * (U_sim - U) @ self._compute_W1(t, sim)
        new_c2 = c2 - r * (U_sim - U) @ self._compute_W2(t, sim)

        return new_c1, new_c2

    def _dummy_grad_desc(
        self,
        t: float,
        sol: Callable[[float], ndarray],
        sim: Callable[[float], ndarray],
        c1: float,
        c2: float,
        r: float,
    ) -> ndarray:
        """Implement the same API as `_gradient_descent` but do nothing.

        To be used with `iterate` without updating parameters.
        """

        return c1, c2

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

        system = self.system

        U, V = L96.apart(simsol(t), system.I, system.J_sim)

        return V.sum(axis=1) * U / system.μ

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

        system = self.system

        U = simsol(t)[: system.I]

        return -U / system.μ

    def _concatenate_sols(self, Δt, num_iters):
        system = self.system

        thresholds = [i * Δt for i in range(num_iters)]

        # Essentially adapt `np.piecewise` to work when input functions have
        # multidimensional output for one-dimensional input.
        def _concat(x, sols, J):
            xn = len(x)

            conditions = np.array([threshold <= x for threshold in thresholds])
            # For a given threshold, many x are greater than it, which would
            # lead to many function evaluations below. This block replaces all
            # but the last True with False, so that each x is only evaluated by
            # one function.
            last_idx = num_iters - 1 - np.argmax(conditions[::-1], axis=0)
            conditions[:] = False
            conditions[last_idx, np.arange(xn)] = True

            functions = [sol.sol for sol in sols]
            y = np.full((system.I + system.I * J, xn), np.inf)

            for condition, function in zip(conditions, functions):
                if np.any(condition):
                    y[:, condition] = function(x[condition])

            return y

        self.sol = lambda x: _concat(x, self.sols, system.J)
        self.sim = lambda x: _concat(x, self.sims, system.J_sim)
