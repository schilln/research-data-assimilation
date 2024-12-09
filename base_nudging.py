"""
The basic code
- to use the nudging approach to data assimilation to nudge an estimated system
  toward an observed "ground truth" system,
- to use gradient-based methods to estimate optimal parameters, and
- to numerically solve an ODE system using Runge–Kutta 4.

`System` is an abstract base class on which the other class(es) and function(s)
in this file depend. Derived classes should implement the unimplemented methods.

`gradient_descent` and `levenberg_marquardt` are two gradient-based methods to
estimate optimal parameters.

`RK4` numerically solves an ODE implementing `System`.
"""

import jax
from jax import numpy as jnp, lax

jndarray = jnp.ndarray


class System:
    def __init__(
        self,
        μ: float,
        bs: jndarray,
        γs: jndarray,
        cs: jndarray,
        observed_slice: slice,
    ):
        """
        An abstract base class defining some common methods and an interface
        on which other classes and functions can rely when using with derived
        classes.

        Parameters
        ----------
        μ
            Nudging parameter
        bs
            Known parameter values of the "true" system, to be used by the
            nudged system as well
        γs
            Unknown parameter values to be used by the "true" system
        cs
            Estimated parameter values to be used by the nudged system (may or
            may not correspond to `γs`)
        observed_slice
            The slice denoting the observed part of the true and nudged system
            states when nudging in `f`. May use `jnp.s_` to define slice to use.
        """
        self._μ = μ
        self._bs = bs
        self._γs = γs
        self._observed_slice = observed_slice

        self.cs = cs

    def f(
        self,
        cs: jndarray,
        true: jndarray,
        nudged: jndarray,
    ) -> tuple[jndarray, jndarray]:
        """

        This function will be jitted.

        Parameters
        ----------
        cs
            Estimated parameter values to be used by the nudged system
        true
            True system
        nudged
            Nudged system

        Returns
        -------
        truep, nudgedp
            The time derivatives of `true` and `nudged`
        """
        s = self.observed_slice

        nudgedp = self.estimated_ode(cs, nudged)
        nudgedp = nudgedp.at[s].subtract(self.μ * (nudged[s] - true[s]))

        return self.ode(true), nudgedp

    def ode(
        self,
        true: jndarray,
    ) -> jndarray:
        """

        This function will be jitted.

        Parameters
        ----------
        true
            True system

        Returns
        -------
        truep
            The time derivative of `true`
        """
        raise NotImplementedError()

    def estimated_ode(
        self,
        cs: jndarray,
        nudged: jndarray,
    ) -> jndarray:
        """

        This function will be jitted.

        Parameters
        ----------
        cs
            Estimated parameter values to be used by the nudged system
        nudged
            Nudged system

        Returns
        -------
        nudgedp
            The time derivative of `nudged`
        """
        raise NotImplementedError()

    def compute_w(self, nudged: jndarray) -> jndarray:
        """Compute the leading-order approximation of the sensitivity equations.

        Subclasses may override this method to optimize computation or to obtain
        higher-order approximations.

        Note this differs from Josh's paper, equation (2.23), by a negative
        sign.

        Parameters
        ----------
        nudged
            The nudged system

        Returns
        -------
        W
            The ith row corresponds to the asymptotic approximation of the ith
            senstitivity corresponding to the ith unknown parameter ci
        """
        # TODO: This requires computing the entire derivative and then slicing
        # it using `observed_slice`. Is it possible to reverse this order to
        # increase efficiency.
        return (
            jax.jacrev(self.estimated_ode, 0)(self.cs, nudged)[
                self.observed_slice
            ].T
            / self.μ
        )

    # The following attributes are read-only.
    μ = property(lambda self: self._μ)
    bs = property(lambda self: self._bs)
    γs = property(lambda self: self._γs)
    observed_slice = property(lambda self: self._observed_slice)


def gradient_descent(
    system: System,
    observed_true: jndarray,
    nudged: jndarray,
    r: float = 1e-4,
):
    """
    Parameters
    ----------
    observed_true
        The observed part of the true system
    nudged
        The whole nudged system
    r
        Learning rate

    Returns
    -------
    new_cs
        New parameter values cs
    """

    diff = nudged[system.observed_slice].ravel() - observed_true.ravel()
    gradient = diff @ system.compute_w(nudged).T

    return system.cs - r * gradient


def levenberg_marquardt(
    system: System,
    observed_true: jndarray,
    nudged: jndarray,
    r: float = 1e-3,
    λ: float = 1e-2,
):
    """
    Parameters
    ----------
    observed_true
        The observed part of the true system.
    nudged
        The whole nudged system
    r
        Learning rate
    λ
        Levenberg–Marquardt parameter

    Returns
    -------
    new_cs
        New parameter values cs
    """
    diff = nudged[system.observed_slice].ravel() - observed_true.ravel()

    gradient = diff @ system.compute_w(nudged).T
    mat = jnp.outer(gradient, gradient)

    step = jnp.linalg.solve(mat + λ * jnp.eye(len(gradient)), gradient)
    return system.cs - r * step


class RK4:
    def __init__(self, system: System):
        f = system.f

        def step(i, vals):
            """This function will be jitted."""
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

        self.step = step

    def solve(
        self,
        system: System,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray, jndarray]:
        tls = jnp.arange(t0, tf, dt)
        N = len(tls)

        # Store the solution at every step.
        true = jnp.full((N, *true0.shape), jnp.inf)
        nudged = jnp.full((N, *nudged0.shape), jnp.inf)

        # Set initial state.
        true = true.at[0].set(true0)
        nudged = nudged.at[0].set(nudged0)

        (true, nudged), _ = lax.fori_loop(
            1, N, self.step, ((true, nudged), (dt, system.cs))
        )

        return true, nudged
