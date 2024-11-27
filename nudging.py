import numpy as np
from jax import numpy as jnp

jndarray = jnp.ndarray

SEED = 42


def simulate_separate(system, solver, t0, tf, dt, interp_method=None):
    I, J, J_sim = system.I, system.J, system.J_sim

    # Initial true state
    init = np.random.default_rng(SEED).uniform(size=I + I * J)

    u0 = jnp.array(init[:I])
    v0 = jnp.array(jnp.reshape(init[I:], (I, J)))

    # Initial simulation state
    u0_sim = jnp.zeros_like(u0)
    v0_sim = jnp.zeros((I, J_sim))

    U, V = solver.solve(
        system,
        u0,
        v0,
        t0,
        tf,
        dt,
    )

    # Nudged solution
    if interp_method is None:
        Un, Vn = solver.solve(
            system,
            u0_sim,
            v0_sim,
            t0,
            tf,
            dt,
            U,
        )
    else:
        Un, Vn = solver.solve(
            system,
            u0_sim,
            v0_sim,
            t0,
            tf,
            dt,
            U,
            interp_method,
        )

    return U, V, Un, Vn


def simulate_simultaneous(system, solver, t0, tf, dt):
    I, J, J_sim = system.I, system.J, system.J_sim

    # Initial true state
    init = np.random.default_rng(SEED).uniform(size=I + I * J)

    u0 = jnp.array(init[:I])
    v0 = jnp.array(jnp.reshape(init[I:], (I, J)))

    # Initial simulation state
    u0_sim = jnp.zeros_like(u0)
    v0_sim = jnp.zeros((I, J_sim))

    U, V, Un, Vn = solver.solve(
        system,
        u0,
        v0,
        u0_sim,
        v0_sim,
        t0,
        tf,
        dt,
    )

    return U, V, Un, Vn


def plot_error(
    fig,
    axs,
    t0: float,
    tf: float,
    dt: float,
    U_true: jndarray,
    V_true: jndarray,
    U_nudge: jndarray,
    V_nudge: jndarray,
):
    """Plot the absolute and relative errors of the nudged solution from the
    true solution.
    """
    tls = np.linspace(t0, tf, len(U_true))

    U_err = np.linalg.norm(U_true - U_nudge, axis=1)
    V_err = np.linalg.norm(V_true - V_nudge, axis=(1, 2))

    U_norm = np.linalg.norm(U_true, axis=1)
    V_norm = np.linalg.norm(V_true, axis=(1, 2))
    U_rel_err = U_err / U_norm
    V_rel_err = V_err / V_norm

    total_err = np.sqrt(U_err**2 + V_err**2)
    total_norm = np.sqrt(U_norm**2 + V_norm**2)
    total_rel_err = total_err / total_norm

    ax = axs[0, 0]
    ax.plot(tls, U_err, label=f"dt = {dt}")
    ax.set_yscale("log")

    ax.set_xlabel("$t$")
    ax.set_title("U absolute error")
    ax.grid()
    ax.legend()

    ax = axs[0, 1]
    ax.plot(tls, U_rel_err)
    ax.set_yscale("log")

    ax.set_xlabel("$t$")
    ax.set_title("U relative error")
    ax.grid()

    ax = axs[1, 0]
    ax.plot(tls, V_err)
    ax.set_yscale("log")

    ax.set_xlabel("$t$")
    ax.set_title("V absolute error")
    ax.grid()

    ax = axs[1, 1]
    ax.plot(tls, V_rel_err)
    ax.set_yscale("log")

    ax.set_xlabel("$t$")
    ax.set_title("V relative error")
    ax.grid()

    ax = axs[2, 0]
    ax.plot(tls, total_err)
    ax.set_yscale("log")

    ax.set_xlabel("$t$")
    ax.set_title("Total error")
    ax.grid()

    ax = axs[2, 1]
    ax.plot(tls, total_rel_err)
    ax.set_yscale("log")

    ax.set_xlabel("$t$")
    ax.set_title("Total relative error")
    ax.grid()
