import numpy as np

from sim.gbm import gbm


def test_gbm_mean_variance():
    # Test the mean and variance of the GBM process
    S0 = 100
    mu = 0.05
    sigma = 0.2
    T = 1
    dt = 1 / 252
    n_paths = 10000

    # Simulate
    S = gbm(S0, mu, sigma, T, dt, n_paths)

    # Theoretical mean and variance
    mean = S0 * np.exp(mu * T)
    variance = S0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)

    # Final values
    final_value = S.iloc[-1].values

    # Calculate the empirical mean and variance
    empirical_mean = np.mean(final_value)
    empirical_variance = np.var(final_value)

    # Assert that the empirical mean and variance are close to the theoretical values
    assert np.isclose(empirical_mean, mean, rtol=0.05)
    assert np.isclose(empirical_variance, variance, rtol=0.1)
