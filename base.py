"""
The basic code
- to use the nudging approach to data assimilation to nudge an estimated system
  toward an observed "ground truth" system, and
- to use gradient-based methods to estimate optimal parameters.

`System` is an abstract base class on which the other class(es) and function(s)
in this file depend. Derived classes should implement the unimplemented methods.

`gradient_descent` and `levenberg_marquardt` are two gradient-based methods to
estimate optimal parameters.
"""

from functools import partial

import jax
from jax import numpy as jnp


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
        return self._compute_w(self.cs, nudged)

    @partial(jax.jit, static_argnames="self")
    def _compute_w(self, cs: jndarray, nudged: jndarray) -> jndarray:
        return (
            jax.jacrev(self.estimated_ode, 0)(cs, nudged)[self.observed_slice].T
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
