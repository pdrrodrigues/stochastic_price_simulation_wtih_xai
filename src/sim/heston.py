import numpy as np
import pandas as pd


def heston(
    S0: float = 100.0,
    v0: float = 0.04,
    mu: float = 0.0,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.7,
    T: float = 1.0,
    dt: float = 1 / 252,
    n_paths: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate price paths under the Heston stochastic volatility model.

    dS_t = mu * S_t dt + sqrt(v_t) * S_t dW_t^S
    dv_t = kappa * (theta - v_t) dt + xi * sqrt(v_t) dW_t^v
    with corr(dW^S, dW^v) = rho.

    Parameters
    ----------
    S0 : float
        Initial price.
    v0 : float
        Initial variance.
    mu : float
        Drift of the asset.
    kappa : float
        Mean-reversion speed of variance.
    theta : float
        Long-run average variance.
    xi : float
        Volatility of volatility.
    rho : float
        Correlation between price and variance shocks.
    T : float
        Time horizon (in years).
    dt : float
        Time step (in years). Default: 1/252 (daily).
    n_paths : int
        Number of simulated paths.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame of shape (n_steps+1, n_paths),
        each column is one simulated price path.
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)

    # Time vector
    times = np.linspace(0, T, n_steps + 1)

    # Preallocate arrays
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))

    # Initial conditions
    S[0, :] = S0
    v[0, :] = v0

    # Cholesky decomposition for correlated Brownian motions
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)

    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=(2, n_paths))
        dW = L @ Z * np.sqrt(dt)

        # Variance process (Euler-Maruyama)
        v_prev = np.maximum(v[t - 1, :], 0.0)
        dv = kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * dW[1]
        v[t, :] = np.maximum(v_prev + dv, 0.0)  # enforce non-negativity

        # Price process
        S_prev = S[t - 1, :]
        dS = mu * S_prev * dt + np.sqrt(v_prev) * S_prev * dW[0]
        S[t, :] = S_prev + dS

    return pd.DataFrame(S, index=times)
