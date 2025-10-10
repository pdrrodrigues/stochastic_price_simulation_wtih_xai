# Finance-SDE-XAI

This project explores **scientific computing in finance** by simulating price paths with **stochastic differential equations (SDEs)** such as **Geometric Brownian Motion (GBM)** and **Heston volatility model**.

On top of the simulations, we train **machine learning surrogates** to predict future volatility or returns, and use **Explainable AI (XAI)** methods (SHAP, LIME) to understand which features (lags, indicators, etc.) drive the predictions.

The project is designed with **software engineering best practices**:
- Modular code (`src/`)
- Unit tests (`tests/`)
- Pre-commit hooks for linting and formatting
- Continuous Integration (GitHub Actions)
- Full containerization with Docker

---

## 🚀 Features
- Simulate GBM and Heston models to generate synthetic OHLCV price series.
- Build lagged and technical features for volatility forecasting.
- Fit baselines (Naïve, EWMA, GARCH) and ML surrogates (Random Forest, XGBoost, MLP).
- Apply SHAP and LIME to explain surrogate predictions.
- Run everything inside a reproducible Docker development container.

---

## 🛠️ Getting Started
```bash
```
