"""
Portfolio Monte Carlo simulator with historical data
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from arch import arch_model

class PortfolioSimulator:
    def __init__(self, pinecone_client):
        """Initialize portfolio simulator with Pinecone client"""
        self.pinecone_client = pinecone_client
        self.asset_data = None
        self.returns_data = None
        self.correlation_matrix = None
        
    def get_portfolio_data(self) -> Dict:
        """Get loaded portfolio data"""
        return {
            'asset_data': self.asset_data,
            'returns_data': self.returns_data,
            'correlation_matrix': self.correlation_matrix
        }
        
    def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Load all asset data from Pinecone"""
        self.asset_data = self.pinecone_client.fetch_all_assets()
        
        # Calculate returns for each asset
        self.returns_data = {}
        for asset, df in self.asset_data.items():
            self.returns_data[asset] = df['close'].pct_change().dropna()
            
        # Find common date range
        date_ranges = []
        for asset, returns in self.returns_data.items():
            date_ranges.append((returns.index.min(), returns.index.max()))
            
        # Use the most restrictive date range
        start_date = max(dr[0] for dr in date_ranges)
        end_date = min(dr[1] for dr in date_ranges)
        
        # Align all data to common dates
        for asset in self.returns_data:
            self.returns_data[asset] = self.returns_data[asset][start_date:end_date]
            
        # Calculate correlation matrix
        returns_df = pd.DataFrame(self.returns_data)
        self.correlation_matrix = returns_df.corr()
        
        return self.asset_data
    
    def _fit_garch_model(self, returns: pd.Series) -> Tuple[float, float, Dict]:
        """Fit GARCH(1,1) model to returns"""
        try:
            # Scale returns to percentage
            returns_pct = returns * 100
            
            # Fit GARCH(1,1) model
            model = arch_model(returns_pct, vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            
            # Extract parameters
            omega = res.params['omega']
            alpha = res.params['alpha[1]']
            beta = res.params['beta[1]']
            
            # Calculate long-run volatility (percent units)
            long_run_var = omega / (1 - alpha - beta)
            long_run_vol = np.sqrt(long_run_var) / 100  # convert to decimal
            
            return returns.mean(), long_run_vol, {
                'omega': omega,
                'alpha': alpha,
                'beta': beta
            }
        except:
            # Fallback to simple statistics
            return returns.mean(), returns.std(), {}
    
    def _robust_cholesky(self, corr: np.ndarray) -> np.ndarray:
        """Compute a Cholesky factor, adding small diagonal bumps if needed."""
        eps0 = 1e-8
        I = np.eye(corr.shape[0])
        for k in range(1000):
            try:
                return np.linalg.cholesky(corr)
            except np.linalg.LinAlgError:
                corr = ((corr + corr.T) / 2) + (eps0 * (1.1 ** k)) * I
        # Last resort: eigen clip
        eigvals, eigvecs = np.linalg.eigh((corr + corr.T) / 2)
        eigvals = np.clip(eigvals, 1e-8, None)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return np.linalg.cholesky(corr)
    
    def _generate_correlated_returns(self, n_days: int, n_simulations: int,
                                   means: Dict, vols: Dict,
                                   distribution: str = 'normal', student_df: int = 8,
                                   correlation_scale: float = 1.0) -> Dict[str, np.ndarray]:
        """Generate correlated returns using Cholesky decomposition"""
        
        # Order assets consistently
        assets = list(means.keys())
        n_assets = len(assets)
        
        # Create mean and volatility vectors
        mu = np.array([means[asset] for asset in assets])
        sigma = np.array([vols[asset] for asset in assets])
        
        # Get correlation matrix in same order and apply scaling to off-diagonals
        base_corr = np.array([[self.correlation_matrix.loc[a1, a2] 
                               for a2 in assets] for a1 in assets])
        I = np.eye(n_assets)
        scaled_corr = I + correlation_scale * (base_corr - I)
        # Clip to valid range for safety
        scaled_corr = np.clip(scaled_corr, -0.99, 0.99)
        np.fill_diagonal(scaled_corr, 1.0)
        
        # Cholesky decomposition (robust)
        L = self._robust_cholesky(scaled_corr)
        
        # Generate random returns
        returns = {asset: [] for asset in assets}
        for _ in range(n_simulations):
            # Generate independent draws
            if distribution == 'student':
                t_samples = stats.t.rvs(df=student_df, size=(n_days, n_assets))
                # Standardize to unit variance: Var(t) = df/(df-2)
                scale = np.sqrt(student_df / (student_df - 2)) if student_df > 2 else 1.0
                z = t_samples / scale
            else:
                z = np.random.randn(n_days, n_assets)
            
            # Apply correlation structure
            correlated_z = z @ L.T
            
            # Scale by volatility and add drift
            for i, asset in enumerate(assets):
                asset_returns = mu[i] + sigma[i] * correlated_z[:, i]
                returns[asset].append(asset_returns)
                
        # Convert to numpy arrays [n_sim, n_days]
        for asset in assets:
            returns[asset] = np.array(returns[asset])
            
        return returns
    
    def run_simulation(self, allocations: Dict[str, float], n_simulations: int = 1000,
                      days_forward: int = 365, initial_value: float = 1000000,
                      risk_free_rate: float = 0.02,
                      rebalance_frequency: str = 'daily',
                      transaction_cost_bps: float = 0.0,
                      distribution: str = 'normal',
                      student_df: int = 8,
                      correlation_scale: float = 1.0) -> Dict:
        """
        Run Monte Carlo simulation for portfolio
        
        Args:
            allocations: Dict of asset allocations (must sum to 100)
            n_simulations: Number of simulation paths
            days_forward: Number of days to simulate
            initial_value: Initial portfolio value
            risk_free_rate: Annual risk-free rate (default 2%)
            rebalance_frequency: 'daily' | 'monthly' | 'quarterly'
            transaction_cost_bps: cost per rebalance as basis points of traded notional (approximate)
            distribution: 'normal' or 'student'
            student_df: degrees of freedom for Student's t (if used)
            correlation_scale: scale for off-diagonal correlations (1.0 = historical)
            
        Returns:
            Dictionary with simulation results
        """
        
        # Load data if not already loaded
        if self.asset_data is None:
            self.load_historical_data()
            
        # Fit models to each asset
        asset_params = {}
        for asset, returns in self.returns_data.items():
            mu, sigma, garch_params = self._fit_garch_model(returns)
            asset_params[asset] = {
                'mu': mu,
                'sigma': sigma,
                'garch': garch_params
            }
            
        # Generate correlated returns
        means = {asset: params['mu'] for asset, params in asset_params.items()}
        vols = {asset: params['sigma'] for asset, params in asset_params.items()}
        
        simulated_returns = self._generate_correlated_returns(
            days_forward, n_simulations, means, vols,
            distribution=distribution, student_df=student_df,
            correlation_scale=correlation_scale
        )
        
        # Normalize allocations to sum to 100
        alloc_copy = allocations.copy()
        total_alloc = sum(alloc_copy.values())
        if total_alloc <= 0:
            raise ValueError("Total allocation must be greater than 0")
        scale = 100.0 / total_alloc
        for k in alloc_copy.keys():
            alloc_copy[k] *= scale
        
        # Determine rebalance period in days
        if rebalance_frequency == 'monthly':
            rebalance_period = 21
        elif rebalance_frequency == 'quarterly':
            rebalance_period = 63
        else:
            rebalance_period = 1
        
        # Transaction cost rate per unit traded
        tc_rate = float(transaction_cost_bps) / 10000.0
        
        assets = list(alloc_copy.keys())
        n_days = days_forward
        S = n_simulations
        
        # Initialize per-asset values [S]
        target_weights = np.array([alloc_copy[a] / 100.0 for a in assets])
        asset_values = {a: np.full(S, initial_value * (alloc_copy[a] / 100.0), dtype=float) for a in assets}
        
        portfolio_values = np.zeros((S, n_days + 1))
        portfolio_values[:, 0] = initial_value
        
        # Simulate day by day with scheduled rebalancing
        for t in range(n_days):
            # Grow each asset by its simulated return for day t
            for i, a in enumerate(assets):
                asset_values[a] = asset_values[a] * (1.0 + simulated_returns[a][:, t])
            
            # Compute portfolio total after growth
            total = np.zeros(S)
            for a in assets:
                total += asset_values[a]
            
            # Rebalance if scheduled
            if (t + 1) % rebalance_period == 0:
                # Current weights per sim
                current_weights = np.stack([asset_values[a] / total for a in assets], axis=1)  # [S, n_assets]
                # Turnover estimate (0.5 * sum |Δw|)
                delta_w = np.abs(current_weights - target_weights)
                turnover = 0.5 * np.sum(delta_w, axis=1)  # [S]
                # Transaction costs
                costs = tc_rate * turnover * total  # [S]
                total_after_costs = np.maximum(0.0, total - costs)
                # Re-allocate to target weights
                for i, a in enumerate(assets):
                    asset_values[a] = target_weights[i] * total_after_costs
                total = total_after_costs
            
            portfolio_values[:, t + 1] = total
        
        # Metrics
        final_values = portfolio_values[:, -1]
        total_returns = (final_values / initial_value - 1)
        annual_returns = (final_values / initial_value) ** (365 / days_forward) - 1
        
        # Drawdowns (mean across paths)
        drawdowns = []
        for path in portfolio_values:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
            drawdowns.append(abs(max_drawdown))
        
        # Annual VaR95 (positive % loss)
        var_95 = abs(np.percentile(annual_returns * 100, 5))
        
        # Sharpe ratio with configurable risk-free rate
        excess_returns = annual_returns - risk_free_rate
        sharpe_denom = np.std(excess_returns)
        sharpe_ratio = np.mean(excess_returns) / sharpe_denom if sharpe_denom > 0 else 0.0
        
        # Recovery time (average days to recover from drawdown)
        recovery_times = []
        for path in portfolio_values:
            running_max = np.maximum.accumulate(path)
            drawdown_start = None
            for i in range(1, len(path)):
                if path[i] < running_max[i] and drawdown_start is None:
                    drawdown_start = i
                elif path[i] >= running_max[i] and drawdown_start is not None:
                    recovery_times.append(i - drawdown_start)
                    drawdown_start = None
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        return {
            'portfolio_values': portfolio_values,
            'final_values': final_values,
            'annual_returns': annual_returns,
            'expected_return': np.mean(annual_returns) * 100,
            'median_final_value': np.median(final_values),
            'best_case_return': np.percentile(annual_returns, 95) * 100,
            'best_case_value': np.percentile(final_values, 95),
            'worst_case_return': np.percentile(annual_returns, 5) * 100,
            'worst_case_value': np.percentile(final_values, 5),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': np.mean(drawdowns),
            'var_95': var_95,
            'recovery_days': int(avg_recovery_time),
            'simulation_days': days_forward,
            'n_simulations': n_simulations
        }