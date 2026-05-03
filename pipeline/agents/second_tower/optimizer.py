"""Mean-variance / CVaR portfolio optimizer with CVXPY/OSQP and SciPy fallback."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def optimize_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    prev_weights: np.ndarray | None,
    max_weight: float,
    max_sector_weight: float,
    max_turnover: float,
    sector_map: dict[str, str] | None,
    tickers: list[str],
    *,
    lambda_risk: float = 1.0,
    risk_objective: str = "variance",
    cvar_alpha: float = 0.05,
    returns_matrix: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
    factor_loadings: np.ndarray | None = None,
    max_factor_risk_share: float = 0.30,
) -> tuple[np.ndarray, dict]:
    """Optimize portfolio weights.

    Tries CVXPY/OSQP first, falls back to SciPy SLSQP, then inverse-volatility.

    Args:
        mu: Expected returns vector (N,)
        sigma: Covariance matrix (N, N)
        prev_weights: Previous period weights (N,) or None
        max_weight: Maximum weight per asset
        max_sector_weight: Maximum total weight per sector
        max_turnover: Maximum total turnover (sum of |w_new - w_old|)
        sector_map: Dict of ticker -> sector string
        tickers: List of ticker strings matching mu/sigma ordering
        lambda_risk: Risk aversion coefficient (default 1.0; higher = more conservative)
        risk_objective: "variance" (default) or "cvar"
        cvar_alpha: CVaR confidence level (default 0.05 → 95th percentile)
        returns_matrix: (T, N) historical returns needed for CVaR objective
        turnover_penalty: L2 penalty coefficient on weight changes (0 = disabled)
        factor_loadings: (N, K) factor loading matrix for risk budgeting
        max_factor_risk_share: max fraction of portfolio risk per factor category

    Returns:
        (weights, info_dict) where weights is (N,) array summing to 1
    """
    N = len(mu)

    # Try CVXPY first
    try:
        if risk_objective == "cvar" and returns_matrix is not None:
            weights, info = _optimize_cvxpy_cvar(
                mu, returns_matrix, prev_weights, max_weight,
                max_sector_weight, max_turnover, sector_map, tickers,
                cvar_alpha, turnover_penalty,
                factor_loadings, max_factor_risk_share, sigma,
                lambda_risk=lambda_risk,
            )
        else:
            weights, info = _optimize_cvxpy(
                mu, sigma, prev_weights, max_weight,
                max_sector_weight, max_turnover,
                sector_map, tickers, turnover_penalty,
                factor_loadings, max_factor_risk_share,
                lambda_risk=lambda_risk,
            )
        if weights is not None:
            return weights, info
    except Exception as e:
        logger.debug(f"CVXPY failed: {e}")

    # Fallback to SciPy
    try:
        weights, info = _optimize_scipy(
            mu, sigma, prev_weights, max_weight, max_turnover, N,
            lambda_risk=lambda_risk,
            sector_map=sector_map, tickers=tickers,
            max_sector_weight=max_sector_weight,
        )
        if weights is not None:
            return weights, info
    except Exception as e:
        logger.debug(f"SciPy fallback failed: {e}")

    # Final fallback: inverse-volatility weighting (strictly better than equal-weight)
    logger.warning("All optimizers failed, using inverse-volatility weights")
    return _inverse_vol_weights(
        sigma, max_weight,
        sector_map=sector_map, tickers=tickers,
        max_sector_weight=max_sector_weight,
    ), {"solver": "inverse_vol", "status": "fallback"}


def _inverse_vol_weights(
    sigma: np.ndarray,
    max_weight: float,
    sector_map: dict[str, str] | None = None,
    tickers: list[str] | None = None,
    max_sector_weight: float = 1.0,
) -> np.ndarray:
    """Inverse-volatility weighting: w_i = (1/sigma_i) / sum(1/sigma_j)."""
    vols = np.sqrt(np.diag(sigma))
    vols = np.where(vols < 1e-10, 1e-10, vols)
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    weights = np.minimum(weights, max_weight)
    s = weights.sum()
    if s > 0:
        weights /= s

    # Enforce sector constraints iteratively
    if sector_map and tickers and max_sector_weight < 1.0:
        sectors: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            sec = sector_map.get(t, "Unknown")
            sectors.setdefault(sec, []).append(i)
        for _ in range(10):
            violated = False
            for _sec, indices in sectors.items():
                sector_sum = weights[indices].sum()
                if sector_sum > max_sector_weight + 1e-10:
                    violated = True
                    scale = max_sector_weight / sector_sum
                    weights[indices] *= scale
            if not violated:
                break
            s = weights.sum()
            if s > 0:
                weights /= s

    return weights


def _optimize_cvxpy(
    mu, sigma, prev_weights, max_weight, max_sector_weight,
    max_turnover, sector_map, tickers, turnover_penalty=0.0,
    factor_loadings=None, max_factor_risk_share=0.30,
    lambda_risk: float = 1.0,
):
    import cvxpy as cp

    N = len(mu)
    w = cp.Variable(N)

    # Objective: maximize mu'w - lambda_risk * w'Sigma*w
    ret = mu @ w
    risk = cp.quad_form(w, sigma)
    obj_expr = ret - lambda_risk * risk

    # Soft L2 turnover penalty
    if turnover_penalty > 0 and prev_weights is not None:
        obj_expr -= turnover_penalty * cp.sum_squares(w - prev_weights)

    objective = cp.Maximize(obj_expr)

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
    ]

    # Sector constraints — applied to all sectors (including single-stock ones)
    if sector_map and tickers:
        sectors = {}
        for i, t in enumerate(tickers):
            s = sector_map.get(t, "Unknown")
            sectors.setdefault(s, []).append(i)
        for sector, indices in sectors.items():
            constraints.append(cp.sum(w[indices]) <= max_sector_weight)

    # Turnover constraint (hard, only when prev_weights available)
    if prev_weights is not None and turnover_penalty == 0:
        constraints.append(cp.norm(w - prev_weights, 1) <= max_turnover)

    # Factor risk budgeting constraints
    if factor_loadings is not None and factor_loadings.shape[0] == N:
        _add_factor_risk_constraints(constraints, w, sigma, factor_loadings, max_factor_risk_share, cp)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=10000)
    except cp.SolverError:
        prob.solve(solver=cp.SCS, verbose=False, max_iter=10000)

    if prob.status in ("optimal", "optimal_inaccurate"):
        weights = np.array(w.value).flatten()
        weights = np.maximum(weights, 0)
        weights = np.minimum(weights, max_weight)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(N) / N
        return weights, {"solver": "cvxpy", "status": prob.status}

    return None, {"solver": "cvxpy", "status": prob.status}


def _optimize_cvxpy_cvar(
    mu, returns_matrix, prev_weights, max_weight,
    max_sector_weight, max_turnover, sector_map, tickers,
    alpha=0.05, turnover_penalty=0.0,
    factor_loadings=None, max_factor_risk_share=0.30, sigma=None,
    lambda_risk: float = 1.0,
):
    """CVaR optimization using Rockafellar-Uryasev LP formulation.

    minimize  -mu'w + lambda_risk * CVaR_alpha(w)
    CVaR_alpha = z + 1/(T*(1-alpha)) * sum(max(-R_t'w - z, 0))
    """
    import cvxpy as cp

    T, N = returns_matrix.shape
    w = cp.Variable(N)
    z = cp.Variable()      # VaR auxiliary variable
    u = cp.Variable(T)     # auxiliary for max(0, ...)

    # CVaR
    cvar = z + (1.0 / (T * (1 - alpha))) * cp.sum(u)

    # Risk-return trade-off with configurable lambda_risk
    obj_expr = mu @ w - lambda_risk * cvar

    # Soft L2 turnover penalty
    if turnover_penalty > 0 and prev_weights is not None:
        obj_expr -= turnover_penalty * cp.sum_squares(w - prev_weights)

    objective = cp.Maximize(obj_expr)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        u >= 0,
        u >= -returns_matrix @ w - z,
    ]

    # Sector constraints — applied to all sectors (including single-stock ones)
    if sector_map and tickers:
        sectors = {}
        for i, t in enumerate(tickers):
            s = sector_map.get(t, "Unknown")
            sectors.setdefault(s, []).append(i)
        for sector, indices in sectors.items():
            constraints.append(cp.sum(w[indices]) <= max_sector_weight)

    if prev_weights is not None and turnover_penalty == 0:
        constraints.append(cp.norm(w - prev_weights, 1) <= max_turnover)

    # Factor risk budgeting
    if factor_loadings is not None and sigma is not None and factor_loadings.shape[0] == N:
        _add_factor_risk_constraints(constraints, w, sigma, factor_loadings, max_factor_risk_share, cp)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iter=10000)
    except cp.SolverError:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=10000)

    if prob.status in ("optimal", "optimal_inaccurate"):
        weights = np.array(w.value).flatten()
        weights = np.maximum(weights, 0)
        weights = np.minimum(weights, max_weight)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(N) / N
        return weights, {"solver": "cvxpy_cvar", "status": prob.status}

    return None, {"solver": "cvxpy_cvar", "status": prob.status}


def _add_factor_risk_constraints(constraints, w, sigma, factor_loadings, max_share, cp):
    """Add per-category factor risk contribution constraints.

    Risk contribution of factor category k: RC_k = w' * B_k * Sigma * B_k' * w
    Constraint: RC_k <= max_share * w' * Sigma * w

    Note: Uses diag(b_k) as a heuristic approximation for the factor projection
    matrix.  The exact decomposition requires an explicit factor covariance
    matrix F (from a factor model Sigma = B @ F @ B' + D), which is not
    available here.  The diag(b_k) approach overestimates per-factor risk
    contribution and is conservative — it serves the practical goal of
    preventing any single factor category from dominating portfolio risk.
    """
    N, K = factor_loadings.shape
    total_risk = cp.quad_form(w, sigma)
    for k in range(K):
        b_k = factor_loadings[:, k]
        # Outer product projection: B_k * Sigma * B_k'
        # Simplified: contribution via factor k loading
        B_k = np.diag(b_k)
        factor_sigma = B_k @ sigma @ B_k
        # Ensure PSD
        factor_sigma = (factor_sigma + factor_sigma.T) / 2
        factor_sigma += 1e-8 * np.eye(N)
        factor_risk = cp.quad_form(w, factor_sigma)
        constraints.append(factor_risk <= max_share * total_risk)


def _optimize_scipy(
    mu, sigma, prev_weights, max_weight, max_turnover, N,
    lambda_risk: float = 1.0,
    sector_map: dict[str, str] | None = None,
    tickers: list[str] | None = None,
    max_sector_weight: float = 1.0,
):
    from scipy.optimize import minimize

    def neg_mv_utility(w):
        port_ret = w @ mu
        port_var = w @ sigma @ w
        return -(port_ret - lambda_risk * port_var)

    bounds = [(0, max_weight)] * N
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if prev_weights is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: max_turnover - np.sum(np.abs(w - prev_weights)),
        })

    # Sector constraints — applied to all sectors (including single-stock ones)
    if sector_map and tickers and max_sector_weight < 1.0:
        sectors: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            sec = sector_map.get(t, "Unknown")
            sectors.setdefault(sec, []).append(i)
        for _sec, indices in sectors.items():
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=indices: max_sector_weight - np.sum(w[idx]),
            })

    x0 = np.ones(N) / N
    result = minimize(neg_mv_utility, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-10})

    if result.success:
        weights = result.x
        weights = np.maximum(weights, 0)
        weights = np.minimum(weights, max_weight)
        if weights.sum() > 0:
            weights /= weights.sum()
        return weights, {"solver": "scipy_slsqp", "status": "success"}

    return None, {"solver": "scipy_slsqp", "status": result.message}
