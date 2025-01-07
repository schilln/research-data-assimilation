"""Gradient-based methods to estimate optimal parameters for the nudged system
in an instance of `base_system.System`."""

from jax import numpy as jnp

from base_system import System

jndarray = jnp.ndarray


def gradient_descent(
    system: System,
    observed_true: jndarray,
    nudged: jndarray,
    r: float = 1e-4,
):
    """Compute one step of gradient descent to obtain optimal parameters for the
    nudged system in `system`.

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
    """Compute one step of Levenberg–Marquardt to obtain optimal parameters for
    thenudged system in `system`.

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
