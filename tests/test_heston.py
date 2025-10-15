import numpy as np
import pandas as pd

from sim.heston import heston


def test_heston():
    # Test the metrics of Heston simulation
    S0 = 100.0
    v0 = 0.04
    mu = 0.05
    kappa = 2.0
    theta = 0.04
    xi = 0.5
    rho = -0.7
    T = 1.0
    dt = 1.0 / 252.0
    n_paths = 10000

    S = heston(S0, v0, mu, kappa, theta, xi, rho, T, dt, n_paths)

    # Test the positivity of simulated stock prices
    assert np.all(S > 0)

    # Test the mean of simulated stock prices
    quantiles = [0.25, 0.5, 0.75, 1.0]
    times = S.index.unique()

    mean = np.zeros(len(quantiles))
    times_quantiles = np.zeros(len(quantiles))

    for i, q in enumerate(quantiles):
        idx = int(q * len(times) - 1) if q < 1.0 else len(times) - 1
        time_point = times[idx]
        times_quantiles[i] = time_point
        mean[i] = S.loc[time_point].mean()

    expected_mean = S0 * np.exp(mu * times_quantiles)
    assert np.allclose(mean, expected_mean, rtol=0.05)

    # Test the martingale property of Heston simulation
    martingale = np.exp(-mu * times_quantiles) * mean
    assert np.allclose(martingale, S0, rtol=0.05)

    # Test autocorrelation of simulated stock prices
    log_returns = S.transform(np.log).diff().dropna()
    autocorr = log_returns.corrwith(log_returns.shift(1)).dropna().mean()
    assert np.allclose(autocorr, 0.0, atol=0.05)

    # Test the sign of simulated stock prices
    assert np.all(np.sign(np.log(S.iloc[-1, :].mean()) - np.log(S0)) == np.sign(mu * T))
