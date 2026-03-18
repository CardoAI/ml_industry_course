"""
MLE for:
    f_k(theta) = ∫ [ Π_{j=1..C} binom(m_jk, s_jk) * p_j(y;theta)^{s_jk} * (1-p_j(y;theta))^{m_jk-s_jk} ] * φ(y) dy

with (Gaussian one-factor / Vasicek form):
    p_j(y; rho_j) = Φ( (Φ^{-1}(PD_j) - sqrt(rho_j)*y) / sqrt(1-rho_j) )

- Integral is evaluated ONLY by discretization on an evenly-spaced grid (simple Riemann sum).
- Optimization is numerical (scipy.optimize.minimize).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, gammaln
from scipy.stats import norm
import json
from dotenv import dotenv_values


# -----------------------------
# Helpers
# -----------------------------
def log_binom(n, k):
    """log( n choose k )"""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def p_vasicek(y, PD, rho):
    """
    Vectorized Vasicek conditional PD.

    y   : (Ny,) grid
    PD  : (C,) unconditional PDs for categories
    rho : (C,) asset correlations in (0,1)
    returns p : (Ny, C)
    """
    y = y[:, None]  # (Ny,1)
    PD = PD[None, :]  # (1,C)
    rho = rho[None, :]  # (1,C)

    a = norm.ppf(PD)
    num = a - np.sqrt(rho) * y
    den = np.sqrt(1.0 - rho)
    return norm.cdf(num / den)


# -----------------------------
# Core log-likelihood
# -----------------------------
def loglik_discretized(rho, S, M, PD, y_grid, dy, shared_rho=False):
    """
    Computes ℓ(rho) = Σ_k log f_k(rho) with f_k integral approximated by Riemann sum.

    S, M: shape (K,C)
    PD  : shape (C,)
    rho : shape (C,) if shared_rho=False, else shape (1,)
    """
    K, C = S.shape

    # Expand rho if it's shared across categories
    if shared_rho:
        rho = np.full(C, float(rho[0]))
    else:
        rho = np.asarray(rho, dtype=float)

    # enforce numerical safety
    eps = 1e-12
    rho = np.clip(rho, eps, 1.0 - eps)

    # precompute y-dependent pieces shared across k
    # p(y): (Ny,C)
    p = p_vasicek(y_grid, PD, rho)
    p = np.clip(p, eps, 1.0 - eps)
    logp = np.log(p)  # (Ny,C)
    log1mp = np.log1p(-p)  # (Ny,C)

    # log φ(y): (Ny,)
    logphi = norm.logpdf(y_grid)  # (Ny,)

    # constant per k: sum_j log binom(m_jk,s_jk)
    const_k = np.sum(log_binom(M, S), axis=1)  # (K,)

    # For each k, compute log ∫ exp( const_k + Σ_j[s logp + (m-s)log(1-p)] + logphi ) dy
    # Vectorize over k and y:
    # term(y,k) = const_k[k] + Σ_j (...) + logphi[y]
    # Build Σ_j (...) as: (Ny,K) via matrix products
    # Σ_j s_jk*logp(y,j) = logp(y,:) @ S[k,:]^T  -> (Ny,K) if we do logp @ S.T
    part1 = logp @ S.T  # (Ny,K)
    part2 = log1mp @ (M - S).T  # (Ny,K)
    log_integrand = part1 + part2 + logphi[:, None] + const_k[None, :]  # (Ny,K)

    # Riemann sum: f_k ≈ Σ_y exp(log_integrand[y,k]) * dy
    log_fk = logsumexp(log_integrand, axis=0) + np.log(dy)  # (K,)

    return np.sum(log_fk)  # scalar


def nll_discretized(params, S, M, PD, y_grid, dy, shared_rho=False):
    """Negative log-likelihood for minimization."""
    ll = loglik_discretized(params, S, M, PD, y_grid, dy, shared_rho=shared_rho)
    return -ll


# -----------------------------
# Fit function
# -----------------------------
def fit_mle(S, M, PD, shared_rho=False, y_max=8.0, n_grid=4001):
    """
    Returns (rho_hat, loglik_hat, scipy_result)

    Discretization:
        y_grid = linspace(-y_max, y_max, n_grid)
        dy = constant spacing
    """
    S = np.asarray(S, dtype=int)
    M = np.asarray(M, dtype=int)
    PD = np.asarray(PD, dtype=float)

    assert S.shape == M.shape
    K, C = S.shape
    assert PD.shape == (C,)

    # y-grid (simple uniform discretization; NOT quadrature routines)
    y_grid = np.linspace(-y_max, y_max, n_grid)
    dy = y_grid[1] - y_grid[0]

    # initialization + bounds
    eps = 1e-6
    if shared_rho:
        x0 = np.array([0.10])
        bounds = [(eps, 1.0 - eps)]
    else:
        x0 = np.full(C, 0.10)
        bounds = [(eps, 1.0 - eps)] * C

    res = minimize(
        nll_discretized,
        x0=x0,
        args=(S, M, PD, y_grid, dy, shared_rho),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10}
    )

    if not res.success:
        print("WARNING: optimizer did not converge:", res.message)

    # unpack estimate
    if shared_rho:
        rho_hat = float(res.x[0])
        ll_hat = -res.fun
        return rho_hat, ll_hat, res
    else:
        rho_hat = res.x.copy()
        ll_hat = -res.fun
        return rho_hat, ll_hat, res


# -----------------------------
# Example usage (adapt to your data)
# -----------------------------
if __name__ == "__main__":
    config = dotenv_values(".env")
    _path = config['PATH_PHD_COURSE']

    rng = np.random.default_rng(0)
    M = pd.read_parquet(f"{_path}/alive_train.parquet")
    S = pd.read_parquet(f"{_path}/target_train.parquet")
    clusters = list(M.columns)
    M = M.to_numpy()
    S = S.to_numpy()
    PD = (S / M).mean(axis=0)

    # Fit separate rho_j for each category:
    rho_hat, ll_hat, res = fit_mle(S, M, PD, shared_rho=False, y_max=8.0, n_grid=4001)
    model_fit = {}
    for c, p in zip(clusters, rho_hat):
        model_fit[c] = p

    # save
    with open(f"{_path}/copula_model_params.json", "w") as f:
        json.dump(model_fit, f)

    print("rho_hat (per category) =", rho_hat)
    print("loglik_hat =", ll_hat)