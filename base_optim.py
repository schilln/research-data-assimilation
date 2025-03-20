"""Algorithms to estimate optimal parameters for the nudged system in an
instance of `base_system.System`.

Should also work with `separate_base_system.System`.
"""

from collections import Counter

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

    def __call__(self, observed_true, nudged):
        update = self.optimizer(observed_true, nudged)
        return jnp.where(self.mask, update, self.system.cs)

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
