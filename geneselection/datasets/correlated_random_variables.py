import numpy as np


def random_corr_mat(D=10, beta=1):
    """Generate random valid correlation matrix of dimension D.
    Smaller beta gives larger off diagonal correlations (beta > 0)."""

    P = np.zeros([D, D])
    S = np.eye(D)

    for k in range(0, D - 1):
        for i in range(k + 1, D):
            P[k, i] = 2 * np.random.beta(beta, beta) - 1
            p = P[k, i]
            for l in reversed(range(k)):
                p = (
                    p * np.sqrt((1 - P[l, i] ** 2) * (1 - P[l, k] ** 2))
                    + P[l, i] * P[l, k]
                )
            S[k, i] = S[i, k] = p

    p = np.random.permutation(D)
    for i in range(D):
        S[:, i] = S[p, i]
    for i in range(D):
        S[i, :] = S[i, p]
    return S
