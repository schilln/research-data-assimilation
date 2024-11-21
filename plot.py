import jax
from jax import numpy as jnp
import numpy as np

jndarray = jnp.ndarray


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
