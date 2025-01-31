from collections.abc import Callable

import numpy as np


def system(t, state, I, J, g, c, F, d, mu):
    # reshaping the input dependent variables
    u = state[0 : I * (J + 1)]
    v = state[I * (J + 1) : 2 * I * (J + 1)]
    w1 = state[2 * I * (J + 1) : 3 * I * (J + 1)]
    w2 = state[3 * I * (J + 1) :]

    # Define u and v
    ul = u[0:I]
    us = np.reshape(u[I:], (I, J))
    # Create index offsets
    ul_1 = np.concatenate(([ul[-1]], ul[0:-1]))  # u_{i+1}
    ul__1 = np.concatenate((ul[1:], [ul[0]]))  # u_{i-1}
    ul_2 = np.concatenate((ul[-2:], ul[0:-2]))  # u_{i-2}
    # Actual differential Equation
    dul = ul_1 * (ul__1 - ul_2) + np.sum(us, axis=1) * g[0] * ul - g[1] * ul + F
    dus = -np.tile(d, (I, 1)) * us - np.transpose(np.tile(g[0] * ul**2, (J, 1)))

    # Define nudged variables
    vl = v[0:I]
    vs = np.reshape(v[I:], (I, J))
    # Index offsets for nudged variables
    vl_1 = np.concatenate(([vl[-1]], vl[0:-1]))
    vl__1 = np.concatenate((vl[1:], [vl[0]]))
    vl_2 = np.concatenate((vl[-2:], vl[0:-2]))
    # Nudged system
    dvl = (
        vl_1 * (vl__1 - vl_2)
        + np.sum(vs, axis=1) * c[0] * vl
        - c[1] * vl
        + F
        - mu * (vl - ul)
    )
    dvs = -np.tile(d, (I, 1)) * vs - np.transpose(np.tile(c[0] * vl**2, (J, 1)))

    # First Sensitivity Equation
    w1l = w1[0:I]
    w1s = np.reshape(w1[I:], (I, J))
    # Index offsets
    w1l_1 = np.concatenate(([w1l[-1]], w1l[0:-1]))
    w1l__1 = np.concatenate((w1l[1:], [w1l[0]]))
    w1l_2 = np.concatenate((w1l[-2:], w1l[0:-2]))
    # Sensitivity Equation 1
    dw1l = (
        w1l_1 * (vl__1 - vl_2)
        + vl_1 * (w1l__1 - w1l_2)
        + np.sum(vs, axis=1) * vl
        + np.sum(w1s, axis=1) * c[0] * vl
        + np.sum(vs, axis=1) * c[0] * w1l
        - c[1] * w1l
        - mu * w1l
    )
    dw1s = (
        -np.tile(d, (I, 1)) * w1s
        - np.transpose(np.tile(vl**2, (J, 1)))
        - np.transpose(np.tile(2 * c[0] * vl * w1l, (J, 1)))
    )

    # Second Sensitivity Equation
    w2l = w2[0:I]
    w2s = np.reshape(w2[I:], (I, J))
    # Index offset
    w2l_1 = np.concatenate(([w1l[-1]], w1l[0:-1]))
    w2l__1 = np.concatenate((w1l[1:], [w1l[0]]))
    w2l_2 = np.concatenate((w1l[-2:], w1l[0:-2]))
    # Sensitivity Equation 2
    dw2l = (
        w2l_1 * (vl__1 - vl_2)
        + vl_1 * (w2l__1 - w2l_2)
        + np.sum(w2s, axis=1) * c[0] * vl
        + np.sum(vs, axis=1) * c[0] * w2l
        - vl
        - c[1] * w2l
        - mu * w2l
    )
    dw2s = -np.tile(d, (I, 1)) * w2s - np.transpose(
        np.tile(2 * c[0] * vl * w2l, (J, 1))
    )

    return np.concatenate(
        (
            dul,
            np.reshape(dus, (I * J)),
            dvl,
            np.reshape(dvs, (I * J)),
            dw1l,
            np.reshape(dw1s, (I * J)),
            dw2l,
            np.reshape(dw2s, (I * J)),
        )
    )


def system_no_sens(t, state, I, J, g, c, F, d, mu):
    """Don't simulate the sensitivities."""

    # reshaping the input dependent variables
    u = state[0 : I * (J + 1)]
    v = state[I * (J + 1) : 2 * I * (J + 1)]

    # Define u and v
    ul = u[0:I]
    us = np.reshape(u[I:], (I, J))
    # Create index offsets
    ul_1 = np.concatenate(([ul[-1]], ul[0:-1]))  # u_{i+1}
    ul__1 = np.concatenate((ul[1:], [ul[0]]))  # u_{i-1}
    ul_2 = np.concatenate((ul[-2:], ul[0:-2]))  # u_{i-2}
    # Actual differential Equation
    dul = ul_1 * (ul__1 - ul_2) + np.sum(us, axis=1) * g[0] * ul - g[1] * ul + F
    dus = -np.tile(d, (I, 1)) * us - np.transpose(np.tile(g[0] * ul**2, (J, 1)))

    # Define nudged variables
    vl = v[0:I]
    vs = np.reshape(v[I:], (I, J))
    # Index offsets for nudged variables
    vl_1 = np.concatenate(([vl[-1]], vl[0:-1]))
    vl__1 = np.concatenate((vl[1:], [vl[0]]))
    vl_2 = np.concatenate((vl[-2:], vl[0:-2]))
    # Nudged system
    dvl = (
        vl_1 * (vl__1 - vl_2)
        + np.sum(vs, axis=1) * c[0] * vl
        - c[1] * vl
        + F
        - mu * (vl - ul)
    )
    dvs = -np.tile(d, (I, 1)) * vs - np.transpose(np.tile(c[0] * vl**2, (J, 1)))

    return np.concatenate(
        (
            dul,
            np.reshape(dus, (I * J)),
            dvl,
            np.reshape(dvs, (I * J)),
        )
    )


def system_true(t, u, I, J, g, F, d):
    """Only simulate the true system."""

    # reshaping the input dependent variables

    # Define u and v
    ul = u[0:I]
    us = np.reshape(u[I:], (I, J))
    # Create index offsets
    ul_1 = np.concatenate(([ul[-1]], ul[0:-1]))  # u_{i+1}
    ul__1 = np.concatenate((ul[1:], [ul[0]]))  # u_{i-1}
    ul_2 = np.concatenate((ul[-2:], ul[0:-2]))  # u_{i-2}
    # Actual differential Equation
    dul = ul_1 * (ul__1 - ul_2) + np.sum(us, axis=1) * g[0] * ul - g[1] * ul + F
    dus = -np.tile(d, (I, 1)) * us - np.transpose(np.tile(g[0] * ul**2, (J, 1)))

    return np.concatenate(
        (
            dul,
            np.reshape(dus, (I * J)),
        )
    )


def system_sim(t, v, I, J, c, F, d, mu, true_u: Callable[[float], np.ndarray]):
    """Only simulate the nudged (simulated) system."""

    # Define u and v
    u = true_u(t)
    ul = u[0:I]

    # Define nudged variables
    vl = v[0:I]
    vs = np.reshape(v[I:], (I, J))
    # Index offsets for nudged variables
    vl_1 = np.concatenate(([vl[-1]], vl[0:-1]))
    vl__1 = np.concatenate((vl[1:], [vl[0]]))
    vl_2 = np.concatenate((vl[-2:], vl[0:-2]))
    # Nudged system
    dvl = (
        vl_1 * (vl__1 - vl_2)
        + np.sum(vs, axis=1) * c[0] * vl
        - c[1] * vl
        + F
        - mu * (vl - ul)
    )
    dvs = -np.tile(d, (I, 1)) * vs - np.transpose(np.tile(c[0] * vl**2, (J, 1)))

    return np.concatenate(
        (
            dvl,
            np.reshape(dvs, (I * J)),
        )
    )
