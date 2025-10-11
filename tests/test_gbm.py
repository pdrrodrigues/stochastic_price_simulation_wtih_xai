import numpy as np

from ..src.sim.gbm import gbm


def test_gbm_mean_variance():
    # Test the mean and variance of the GBM process
    S0 = 100
    mu = 0.05
    sigma = 0.2
    T = 1
    dt = 1 / 252

    # Simulate
    S = gbm(S0, mu, sigma, T, dt)

    # Theoretical mean and variance
    mean = S0 * np.exp(mu * T)
    variance = S0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)

    # Calculate the empirical mean and variance
    empirical_mean = np.mean(S)
    empirical_variance = np.var(S)

    # Assert that the empirical mean and variance are close to the theoretical values
    assert np.isclose(empirical_mean, mean, rtol=1e-2)
    assert np.isclose(empirical_variance, variance, rtol=5e-2)
