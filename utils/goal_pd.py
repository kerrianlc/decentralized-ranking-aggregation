from numba import njit


@njit
def prox_r(z, a_i, gamma, beta):
    if z < a_i - gamma * beta:
        return z + gamma * beta
    elif z > a_i + gamma:
        return z - gamma
    else:
        return a_i


@njit
def goalpd_update(x, a, mu, degrees, i, j, rho, beta):
    """
    x: (n, m)
    a: (n, m)
    mu: (n, m)
    degrees: (n,)
    i, j: int
    rho: float
    beta: float
    """

    _, m = x.shape

    for k in range(m):
        # edge average
        z_e = 0.5 * (x[i, k] + x[j, k])

        # dual updates
        mu[j, k] += rho * (z_e - x[j, k]) / degrees[j]
        mu[i, k] += rho * (z_e - x[i, k]) / degrees[i]

        rho_i = rho * degrees[i]
        rho_j = rho * degrees[j]

        z_i = z_e + mu[i, k] / rho
        z_j = z_e + mu[j, k] / rho

        # proximal step
        x[i, k] = prox_r(z_i, a[i, k], 1.0 / rho_i, beta)
        x[j, k] = prox_r(z_j, a[j, k], 1.0 / rho_j, beta)
