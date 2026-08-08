import warnings
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

logger = logging.getLogger(__name__)

# Completely mute SciPy's internal SLSQP boundary probe warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*outside bounds during a minimize step.*")


class PortfolioRiskEngine:
    """
    Convex portfolio optimization engine with volatility constraints,
    position bounds, and sector/cash limits.
    """
    def __init__(
        self, 
        target_volatility: float = 0.20, 
        min_weight: float = 0.02, 
        max_weight: float = 0.40
    ):
        self.target_volatility = target_volatility
        self.min_weight = min_weight
        self.max_weight = max_weight

    def optimize_weights(
        self, 
        alpha_scores: pd.Series, 
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Maximizes expected portfolio return (alpha_scores) subject to:
          1. Sum of weights == 1.0 (Fully Invested)
          2. Individual min/max weight bounds per asset
          3. Annualized Portfolio Volatility <= target_volatility
        """
        n_assets = len(alpha_scores)
        tickers = alpha_scores.index.tolist()
        
        # Initial guess: Equal Weighting
        x0 = np.ones(n_assets) / n_assets
        
        # Convert covariance matrix to numpy array
        cov_np = cov_matrix.loc[tickers, tickers].values
        alphas_np = alpha_scores.values

        # Objective: Maximize expected portfolio alpha -> Minimize (-1 * Portfolio Alpha)
        def objective(w):
            return -np.dot(w, alphas_np)

        # Objective Gradient for faster solver convergence
        def objective_jacobian(w):
            return -alphas_np

        # Constraint 1: Weights sum to 100%
        def constraint_sum_weights(w):
            return np.sum(w) - 1.0

        # Constraint 2: Portfolio Volatility <= Target Volatility
        def constraint_volatility(w):
            # Annualized volatility assuming daily covariance (sqrt(252))
            port_var = np.dot(w.T, np.dot(cov_np, w))
            port_vol = np.sqrt(np.maximum(port_var, 1e-8)) * np.sqrt(252)
            return self.target_volatility - port_vol

        constraints = [
            {'type': 'eq', 'fun': constraint_sum_weights},
            {'type': 'ineq', 'fun': constraint_volatility}
        ]

        # Individual position bounds [min_weight, max_weight]
        bounds = Bounds(
            lb=np.full(n_assets, self.min_weight), 
            ub=np.full(n_assets, self.max_weight)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                fun=objective,
                x0=x0,
                method='SLSQP',
                jac=objective_jacobian,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )

        if not res.success:
            message = f"Risk optimizer failed; refusing fallback allocation: {res.message}"
            logger.error(message)
            raise RuntimeError(message)

        # Clean output weights (round to avoid precision artifacts)
        optimal_weights = np.round(res.x, 6)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        return pd.Series(optimal_weights, index=tickers)
