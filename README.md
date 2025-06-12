# Nudging and optimizing systems of differential equations

The following files will be useful to define systems of differential equations, simulate them forward in time, and optimize their parameters.

- `base_system.py`

  Contains an abstract base class `System`.
  It should be subclassed to implement the desired system of differential equations (e.g., Lorenz '63).
  The `System` class provides an interface and some concrete methods to work with
  - instances of `base_solver.Solver` to simulate the desired system forward in time, and
  - functions in `base_optim.py` to estimate optimal parameters of the system.

  Note that some concrete implementations of `base_solver.Solver` are available in `solvers.py`.

- `solvers.py`

  Contains concrete implementations of `base_solver.Solver` to simulate a given instance of `base_system.System` forward in time.

- `base_optim.py`

  Contains gradient-based methods to estimate optimal parameters of an instance of `base_system.System` using a (partially-)observed true system.

- `utils.py`

  Provides a function to iteratively simulate a given instance of `base_system.System` forward in time and optimize parameters at regular intervals.

## Examples

See the folder `examples`.
Specific examples of note:

- `L63.py` and `L63.ipynb`

  The Lorenz '63 system.

- `L96.py` and `L96.ipynb`

  The Lorenz '96 system.
  Note the custom `run_update` (similar to `utils.run_update`) which returns the errors of the large- and small-scale systems separately.

- `L63_modified.ipynb`

  Demonstrates regularization using `base_optim.Regularizer`.

- `harmonic_oscillator.ipynb`

  Demonstrates "pruning," that is, permanently setting to zero parameters that fall below a specified threshold for a specified number of iterations.

## Equation Discovery Given Externally Produced Data

Most of the code in this repository is designed for twin experiments, in which "true" data is produced alongside a "simulated" system whose parameters are to be optimized to match the "true" system.
However, in application the parameter estimation is done on data over which one has no control because it is produced externally.
The files

- `separate_base_system.py`,
- `separate_base_solver.py`,
- `separate_solvers.py`, and
- `separate_utils.py`

constitute beginning applications of this codebase to this type of problem.
For examples, see the folder `separate_examples`.

## Dependencies

See `requirements_jax_gpu.txt` for required Python dependencies when using a GPU, and `requirements.txt` when not using a GPU.

## Creating custom ODE solvers

See `base_solver.py`, which contains some abstract base classes, and `solvers.py`, which contains concrete implementations/subclasses.
Other code, such as that in `utils.py`, should be able to work with new subclasses.

## Creating custom optimizers

See `base_optim.py`, which contains an abstract base class and some concrete implementations/subclasses.
Other code, such as that in `utils.py`, should be able to work with new subclasses.
