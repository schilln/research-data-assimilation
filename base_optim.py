"""Algorithms to estimate optimal parameters for the nudged system in an
instance of `base_system.System`.

Should also work with `separate_base_system.System`.
"""

from collections.abc import Callable
from collections import Counter

import jax
from jax import numpy as jnp
import numpy as np
import optax

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

    def compute_gradient(
        self, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        """Compute the derivative of the error with respect to each parameter,
        as in (2.8) of Josh's thesis.

        Note this differs from (2.8) in that it handles complex-valued true and
        nudged states.
        """
        diff = (
            nudged[self.system.observed_slice].ravel() - observed_true.ravel()
        )
        w = self.system.compute_w(nudged)
        m = w.shape[0]
        gradient = jnp.real(diff.conj() @ w.T.reshape(-1, m))
        return gradient

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
        gradient = self.compute_gradient(observed_true, nudged)

        return -self.learning_rate * gradient


class WeightedLevenbergMarquardt(Optimizer):
    def __init__(
        self, system: System, learning_rate: float = 1e-3, lam: float = 1e-2
    ):
        """Perform a weighted version of the Levenberg–Marquardt modification of
        Gauss–Newton.

        Parameters
        ----------
        learning_rate
            The learning rate (scalar by which to multiply the step)
        lam
            Levenberg–Marquardt parameter
        """
        super().__init__(system)
        self.learning_rate = learning_rate
        self.lam = lam

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        gradient = self.compute_gradient(observed_true, nudged)

        mat = jnp.outer(gradient, gradient)

        step = jnp.linalg.solve(
            mat + self.lam * jnp.eye(mat.shape[0]), gradient
        )

        return -self.learning_rate * step


class LevenbergMarquardt(Optimizer):
    def __init__(
        self, system: System, learning_rate: float = 1e-3, lam: float = 1e-2
    ):
        """Perform the Levenberg–Marquardt modification of Gauss–Newton.

        Parameters
        ----------
        learning_rate
            The learning rate (scalar by which to multiply the step)
        lam
            Levenberg–Marquardt parameter
        """
        super().__init__(system)
        self.learning_rate = learning_rate
        self.lam = lam

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        gradient = self.compute_gradient(observed_true, nudged)

        w = self.system.compute_w(nudged)
        m = w.shape[0]
        w_flat = w.reshape(m, -1)
        mat = jnp.real(w_flat.conj() @ w_flat.T)

        step = jnp.linalg.solve(
            mat + self.lam * jnp.eye(mat.shape[0]), gradient
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
        """Use regularization on the parameters of `System`.

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


class OptaxWrapper(Optimizer):
    def __init__(
        self, system: System, optimizer: optax.GradientTransformationExtraArgs
    ):
        """Wrap a given Optax optimizer.

        Parameters
        ----------
        optimizer
            Instance of `optax.GradientTransformationExtraArgs`
            For example, `optax.adam(learning_rate=1e-1)`.
        """
        super().__init__(system)
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(system.cs)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        gradient = self.compute_gradient(observed_true, nudged)

        update, self.opt_state = self.optimizer.update(gradient, self.opt_state)
        return update


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


def pruned_factory(system_type: type[System]) -> type[System]:
    """Return a 'pruned' variant of `system_type`.

    If a parameter in `cs` of the system is to be set below its corresponding
    threshold (in absolute value), it will be set to zero permanently.
    Optionally require that this occur at least a specified number of times
    consecutively before setting a parameter to zero permanently.
    """

    class Pruned(system_type):
        def __init__(
            self,
            *args,
            threshold: float | jndarray | np.ndarray,
            iterations: int | jndarray | np.ndarray | None = None,
            **kwargs,
        ):
            """

            Parameters
            ----------
            threshold
                If a float, all parameters `cs` are compared against this common
                value.
                If an array, each parameter is compared against the value in
                `cs` in the same position.
                To disable pruning for a parameter, set its threshold to zero.
            iterations
                Require each parameter to be less than its corresponding
                threshold at least `iterations` times consecutively before
                setting it to zero (permanently).
                As with `threshold`, if an `int`, each parameter will be
                use this common value, but if an array, then each parameter will
                use it corresponding value.
                If None, only one time being less than `threshold` is needed
                to set a parameter to zero.
            """
            super().__init__(*args, **kwargs)

            if isinstance(threshold, (jndarray, np.ndarray)):
                if self._cs.shape != threshold.shape:
                    raise ValueError(
                        "`threshold` must have same shape as `system.cs`"
                    )
            self.threshold = np.array(threshold)

            if isinstance(iterations, (jndarray, np.ndarray)):
                if self._cs.shape != iterations.shape:
                    raise ValueError(
                        "`iterations` must have same shape as `system.cs`"
                    )
            self.iterations = (
                None if iterations is None else np.array(iterations)
            )

            # A mask in which True indicates the corresponding parameter should
            # be set to zero.
            self._set_zero = np.zeros_like(self.cs, dtype=bool)

            # Count the number of times each parameter is below its threshold in
            # a row.
            self._counter = np.zeros_like(self.cs, dtype=int)

        def _set_cs(self, cs):
            # For parameters under the threshold, set the mask to True.
            # Don't change the mask where it already was True.
            below_threshold = np.abs(self.cs) < self.threshold

            # Increment the counter to parameters below their threshold and
            # reset to zero the counter for parameters not below their
            # threshold.
            if self.iterations is not None:
                self._counter += below_threshold
                self._counter[~below_threshold] = 0
                at_least_counter = self._counter >= self.iterations
            else:
                at_least_counter = True

            set_zero = below_threshold & at_least_counter
            self._set_zero[set_zero] = True
            self._cs = jnp.where(self._set_zero, 0, cs)

            # Reset the counter for parameters already set to zero (no point
            # continuing to count).
            if self.iterations is not None:
                self._counter[self._set_zero] = 0

        cs = property(
            lambda self: self._cs, lambda self, value: self._set_cs(value)
        )

    doc = (
        "This is a 'Pruned' version of the original class "
        f"({system_type.__module__}.{system_type.__qualname__}); "
        "that is, if a parameter in `self.cs` is to be set below its "
        "corresponding threshold (in absolute value), it will be set to zero "
        "permanently."
    )

    Pruned.__module__ = system_type.__module__
    Pruned.__name__ = f"{system_type.__name__}_Pruned"
    Pruned.__qualname__ = system_type.__qualname__
    Pruned.__doc__ = (
        doc
        if system_type.__doc__ is None
        else system_type.__doc__ + "\n\n" + doc
    )
    Pruned.__annotations__ = system_type.__annotations__

    return Pruned


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
