Josh's paper assumes the two parameters to be estimated are

$$
\gamma_1 = \gamma_{i, j} = \gamma_i \quad \text{for all } i, j\\
\gamma_2 = \bar d_i \quad \text{for all } i
$$

The first parameter $\gamma_1$ corresponds to `γs` and `γs2` which are all identical.

The second parameter $\gamma_2$ corresponds to `ds` which are all identical.

Thus the sensitivities in Josh's paper correspond to these two parameters, and the functions `compute_W1` and `compute_W2` reflect this.
That is, while the function `ode` is general to the Lorenz '96 system, the functions `compute_Wi` are not.
