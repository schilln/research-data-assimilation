"""Algorithms to estimate optimal parameters for the nudged system in an
instance of `base_system.System`.

Should also work with `separate_base_system.System`.
"""

from collections.abc import Callable
from collections import Counter

import jax
from jax import numpy as jnp

from base_system import System

jndarray = jnp.ndarray


class Optimizer:
    def __init__(self, system: System):
        """Abstract base class for optimizers of `System`s to compute updated
        parameter values.

        Subclasses should implement `step`.

        They may optionally override `__init__` (such as to store other
        algorithm parameters as attributes), but should call
        `super().__init__(system)` to properly store `system` as an attribute.

        Parameters
        ----------
        system
            Instance of `System` whose unknown parameters (`system.cs`) are to
            be optimized

        Methods
        -------
        __call__

        Abstract Methods
        ----------------
        step
        """
        self._system = system

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        """Compute the step to take to update the parameters of `system`.

        Parameters
        ----------
        observed_true
            The observed portion of the true system's state
        nudged
            The nudged system's state

        Returns
        -------
        step
            The vector to add to `system.cs` to obtain the new parameters
        """

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
        return self.system.cs + jnp.real(self.step(observed_true, nudged))

    # The following attribute is read-only.
    system = property(lambda self: self._system)


class PartialOptimizer(Optimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        param_idx: jndarray | None = None,
    ):
        """Optimize only specified parameters.

        Parameters
        ----------
        optimizer
            An optimizer that will be used to perform parameter updates,
            ignoring updates to parameters not specified in `param_idx`.
        param_idx
            An array specifying the parameters to be updated. The updates for
            other parameters will be set to zero.

            For example, to update only the first, third, and fourth parameters
            (as determined from the ordering of `system.cs` for a given instance
            of `System`), one would use `np.array([0, 2])`.
        """
        # Define the attributes that
        super().__setattr__(
            "_own_attrs", {"_system", "system", "optimizer", "mask"}
        )
        super().__init__(optimizer.system)

        self.optimizer = optimizer

        n = len(self.optimizer.system.cs)
        self.mask = jnp.zeros(n, dtype=bool)
        self.mask = self.mask.at[param_idx].set(True)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        update = self.optimizer.step(observed_true, nudged)
        return jnp.where(self.mask, update, 0)

    def __getattr__(self, name):
        """For attributes that aren't defined in this class, route access to the
        wrapped optimizer.
        """
        return getattr(self.optimizer, name)

    def __setattr__(self, name, value):
        """For attributes that aren't defined in this class, route access to the
        wrapped optimizer.
        """
        if name in self._own_attrs:
            super().__setattr__(name, value)
        else:
            setattr(self.optimizer, name, value)


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

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        diff = (
            nudged[self.system.observed_slice].ravel() - observed_true.ravel()
        )
        w = self.system.compute_w(nudged)
        gradient = diff @ jnp.reshape(w.T, (-1, w.shape[0]))

        return -self.learning_rate * gradient


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

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        diff = (
            nudged[self.system.observed_slice].ravel() - observed_true.ravel()
        )
        w = self.system.compute_w(nudged)
        gradient = diff @ jnp.reshape(w.T, (-1, w.shape[0]))
        mat = jnp.outer(gradient, gradient)

        step = jnp.linalg.solve(
            mat + self.lam * jnp.eye(len(gradient)), gradient
        )

        return -self.learning_rate * step


class Regularizer(Optimizer):
    def __init__(
        self,
        system: System,
        ord: int | float | Callable,
        prior: jndarray | None = None,
        callable_is_derivative: bool | None = None,
    ):
        """

        Parameters
        ----------
        ord
            If a float, take the regularizing function/penalty on the parameters
            to be the `ord`-norm of the parameters.
            If a callable, see `callable_is_derivative`.
        prior
            The prior expected values of the parameters, i.e., distance of the
            parameters from the prior will be penalized.
            If not given or None, taken to be zero (as in typical
            regularization).
            Must be the same shape as `system.cs`.
        callable_is_derivative
            The following rules apply if `ord` is a callable:
            If True, `ord` should return an array of the same size as the input
            (i.e., as `system.cs`).
            If False, `ord` is taken to be the regularizing function/penalty on
            the parameters, and will be auto-differentiated to compute its
            derivative with respect to the parameters.
        """
        if prior is None:
            self._prior = jnp.zeros_like(system.cs)
        else:
            if prior.shape != system.cs.shape:
                raise ValueError(
                    "`prior` should have same shape as `system.cs`"
                )
            self._prior = prior

        match ord:
            case int() | float():
                pass
            case Callable() if callable_is_derivative is None:
                raise ValueError(
                    "`callable_is_derivative` must be a bool when `ord` is a "
                    "callable"
                )
            case Callable() if callable_is_derivative:
                if ord(system.cs).shape != system.cs.shape:
                    raise ValueError(
                        "`ord` must return an array of the same shape as the "
                        "parameters `system.cs`"
                    )
            case Callable() if not callable_is_derivative:
                if not jnp.isscalar(ord(system.cs)):
                    raise ValueError(
                        "`ord` must be scalar-valued since "
                        "`callable_is_derivative` is False"
                    )
            case _:
                raise ValueError("`ord` is an invalid type")

        super().__init__(system)
        self._ord = ord
        self._callable_is_derivative = callable_is_derivative

    def step(self, *_):
        ord, prior = self.ord, self.prior
        cs = self.system.cs
        match ord:
            case 2:
                return -2 * (cs - prior)
            case 1:
                return -jnp.sign(cs - prior)
            case int() | float():
                # FutureFIXME: Evaluating at `cs - prior` might not be right.
                return -jax.jacfwd(lambda ps: jnp.norm(ps, ord=ord))(cs - prior)
            case Callable() if self.callable_is_derivative:
                # FutureFIXME: Evaluating at `cs - prior` might not be right.
                return -ord(cs - prior)
            case Callable() if not self.callable_is_derivative:
                # FutureFIXME: Evaluating at `cs - prior` might not be right.
                return -jax.jacfwd(ord, holomorphic=True)(cs - prior)
            case _:
                raise ValueError("`self.ord` is no longer a valid value")

    ord = property(lambda self: self._ord)
    callable_is_derivative = property(lambda self: self._callable_is_derivative)
    prior = property(lambda self: self._prior)


class OptimizerChain(Optimizer):
    def __init__(
        self,
        system: System,
        learning_rate: float,
        optimizers: list[Optimizer],
        weights: list[float],
    ):
        """Use several `Optimizer`s together, such as gradient descent with
        regularization.

        Parameters
        ----------
        learning_rate
            The amount by which to scale the total update/step size
        optimizers
            A list of `Optimizer`s whose updates to the parameters of `system`
            will be summed.
            It may be convenient to set the learning rate of each optimizer (if
            available) to one, since this class also uses a learning rate and
            relative weights.
        weights
            The relative weights to place on each optimizer's step. Each weight
            will be divided by the sum of all weights so that sum of the weights
            is one.
        """
        assert len(optimizers) == len(weights), (
            "`optimizers` and `weights` should have same length"
        )

        super().__init__(system)
        self.learning_rate = learning_rate
        self._optimizers = optimizers
        self._weights = jnp.array(weights) / sum(weights)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        return self.learning_rate * sum(
            [
                weight * optimizer.step(observed_true, nudged)
                for weight, optimizer in zip(self.weights, self.optimizers)
            ]
        )

    optimizers = property(lambda self: self._optimizers)
    weights = property(lambda self: self._weights)


class LRScheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError


class DummyLRScheduler(LRScheduler):
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float = 0.99):
        """

        Parameters
        ----------
        optimizer
            An instance of `Optimizer` with a `learning_rate` attribute.
        gamma
            Multiply the learning rate of `optimizer` by `gamma` with every call
            to `step`.
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.optimizer.learning_rate *= self.gamma


class MultiStepLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int] | tuple[int],
        gamma: float = 0.5,
    ):
        """

        Inspired by PyTorch's `MultiStepLR`

        Parameters
        ----------
        optimizer
            An instance of `Optimizer` with a `learning_rate` attribute.
        milestones
            For each milestone, update the learning rate after that many calls
            to `step`.
            Specifying the same milestone m times will result in
            multiplying the learning rate by `gamma` m times at that milestone.
        gamma
            Multiply the learning rate of `optimizer` by `gamma` upon reaching
            each milestone in `milestones`.
        """
        super().__init__(optimizer)
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.steps = 0

    def step(self):
        self.steps += 1
        if self.steps in self.milestones:
            self.optimizer.learning_rate *= (
                self.gamma ** self.milestones[self.steps]
            )
