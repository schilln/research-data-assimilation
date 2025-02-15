"""Algorithms to estimate optimal parameters for the nudged system in an
instance of `base_system.System`.

Should also work with `separate_base_system.System`.
"""

from jax import numpy as jnp

from base_system import System

jndarray = jnp.ndarray


class Optimizer:
    def __init__(self, system: System):
        """Abstract base class for optimizers of `System`s to compute updated
        parameter values.

        Subclasses should implement `__call__`.

        They may optionally override `__init__` (such as to store other
        algorithm parameters as attributes), but should call
        `super().__init__(system)` to properly store `system` as an attribute.

        Parameters
        ----------
        system
            Instance of `System` whose unknown parameters (`system.cs`) are to
            be optimized

        Abstract Methods
        ----------------
        __call__
        """
        self._system = system

    def __call__(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        """Compute the new parameter values following one step of the
        optimization algorithm.

        Parameters
        ----------
        observed_true
            The observed portion of the true system's state
        nudged
            The nudged system's state

        Returns
        -------
        new_cs
            The new values for `system.cs`
        """
        raise NotImplementedError()

    # The following attribute is read-only.
    system = property(lambda self: self._system)


class GradientDescent(Optimizer):
    def __init__(self, system: System, learning_rate: float = 1e-4):
        """Perform gradient descent.

        See documentation of `Optimizer`.

        Parameters
        ----------
        learning_rate
            The learning rate to use in gradient descent
        """

        super().__init__(system)
        self.learning_rate = learning_rate

    def __call__(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        diff = (
            nudged[self.system.observed_slice].ravel() - observed_true.ravel()
        )
        gradient = diff @ self.system.compute_w(nudged).T

        return self.system.cs - self.learning_rate * gradient


class LevenbergMarquardt(Optimizer):
    def __init__(
        self, system: System, learning_rate: float = 1e-3, lam: float = 1e-2
    ):
        """Perform the Levenberg–Marquardt algorithm of optimization.

        Parameters
        ----------
        learning_rate
            The learning rate to use in gradient descent
        lam
            Levenberg–Marquardt parameter
        """
        super().__init__(system)
        self.learning_rate = learning_rate
        self.lam = lam

    def __call__(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        diff = (
            nudged[self.system.observed_slice].ravel() - observed_true.ravel()
        )

        gradient = diff @ self.system.compute_w(nudged).T
        mat = jnp.outer(gradient, gradient)

        step = jnp.linalg.solve(
            mat + self.lam * jnp.eye(len(gradient)), gradient
        )
        return self.system.cs - self.learning_rate * jnp.real(step)
