import numpy as np
import pandas as pd


def gbm(
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    dt: float = 1 / 252,
    n_paths: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulates price of asset using Geometric Brownian Motion (GBM) with the exact solution

    Parameters
    ----------
    S0: float
        Initial asset price
    mu: float
        Drift coefficient
    sigma: float
        Volatility coefficient
    T: float
        Time horizon
    dt: float
        Time step
    n_paths: int
        Number of paths to simulate
    seed: int | None
        Seed for random number generator

    Returns
    -------
    - pd.DataFrame
        Wide-format DataFrame of shape (n_steps+1, n_paths)
        each column represents a path of the asset price over time.
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)

    # Time vector
    times = np.linspace(0, T, n_steps + 1)

    # Preallocate array
    S = np.zeros((n_steps + 1, n_paths))

    # Initial condition
    S[0, :] = S0

    # Simulate asset price
    for i in range(1, n_steps + 1):
        S[i, :] = S[i - 1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * np.random.normal(size=n_paths)
        )

    return pd.DataFrame(S, index=times)
