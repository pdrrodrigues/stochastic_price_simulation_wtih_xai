import numpy as np


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

    assert np.all(S > 0)
    quantiles = [0.25, 0.5, 0.75, 1.0]
    times = S.index.unique()

    mean_values = np.zeros(len(quantiles))
    times_quantiles = np.zeros(len(quantiles))

    for i, q in enumerate(quantiles):
        idx = int(q * len(times) - 1) if q < 1.0 else len(times) - 1
        time_point = times[idx]
        times_quantiles[i] = time_point
        mean_values[i] = S.loc[time_point].mean()

    expected_mean = S0 * np.exp(mu * times_quantiles)
    assert np.allclose(mean_values, expected_mean, rtol=0.05)
