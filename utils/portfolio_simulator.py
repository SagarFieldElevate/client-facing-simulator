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
            
            # Calculate long-run volatility
            long_run_var = omega / (1 - alpha - beta)
            long_run_vol = np.sqrt(long_run_var) / 100  # Convert back to decimal
            
            return returns.mean(), long_run_vol, {
                'omega': omega,
                'alpha': alpha,
                'beta': beta
            }
        except:
            # Fallback to simple statistics
            return returns.mean(), returns.std(), {}
    
    def get_optimal_allocation_for_sharpe(self, base_allocations: Dict[str, float], 
                                         crypto_range: List[float] = None,
                                         days_forward: int = 365,
                                         n_simulations: int = 1000) -> Tuple[float, float]:
        """
        Find optimal crypto allocation that maximizes Sharpe ratio
        
        Args:
            base_allocations: Base allocations without crypto
            crypto_range: List of crypto percentages to test
            days_forward: Simulation period
            n_simulations: Number of simulations
            
        Returns:
            Tuple of (optimal_crypto_percentage, max_sharpe_ratio)
        """
        if crypto_range is None:
            crypto_range = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
        
        best_crypto = 0
        best_sharpe = -np.inf
        
        for crypto_pct in crypto_range:
            # Adjust allocations
            test_allocations = base_allocations.copy()
            
            # Scale down other allocations
            scale_factor = (100 - crypto_pct) / 100
            for asset in test_allocations:
                test_allocations[asset] *= scale_factor
            test_allocations['crypto'] = crypto_pct
            
            # Run simulation
            results = self.run_simulation(
                allocations=test_allocations,
                n_simulations=n_simulations,
                days_forward=days_forward,
                initial_value=1000000  # Use standard value
            )
            
            if results['sharpe_ratio'] > best_sharpe:
                best_sharpe = results['sharpe_ratio']
                best_crypto = crypto_pct
        
        return best_crypto, best_sharpe
    
    def _generate_correlated_returns(self, n_days: int, n_simulations: int,
                                   means: Dict, vols: Dict) -> Dict[str, np.ndarray]:
        """Generate correlated returns using Cholesky decomposition"""
        
        # Order assets consistently
        assets = list(means.keys())
        n_assets = len(assets)
        
        # Create mean and volatility vectors
        mu = np.array([means[asset] for asset in assets])
        sigma = np.array([vols[asset] for asset in assets])
        
        # Get correlation matrix in same order
        corr_matrix = np.array([[self.correlation_matrix.loc[a1, a2] 
                                for a2 in assets] for a1 in assets])
        
        # Cholesky decomposition
        L = np.linalg.cholesky(corr_matrix)
        
        # Generate random returns
        returns = {}
        for sim in range(n_simulations):
            # Generate independent standard normal random variables
            z = np.random.randn(n_days, n_assets)
            
            # Apply correlation structure
            correlated_z = z @ L.T
            
            # Scale by volatility and add drift
            for i, asset in enumerate(assets):
                if asset not in returns:
                    returns[asset] = []
                    
                asset_returns = mu[i] + sigma[i] * correlated_z[:, i]
                returns[asset].append(asset_returns)
                
        # Convert to numpy arrays
        for asset in assets:
            returns[asset] = np.array(returns[asset])
            
        return returns
    
    def run_simulation(self, allocations: Dict[str, float], n_simulations: int = 1000,
                      days_forward: int = 365, initial_value: float = 1000000) -> Dict:
        """
        Run Monte Carlo simulation for portfolio
        
        Args:
            allocations: Dict of asset allocations (must sum to 100)
            n_simulations: Number of simulation paths
            days_forward: Number of days to simulate
            initial_value: Initial portfolio value
            
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
            days_forward, n_simulations, means, vols
        )
        
        # Calculate portfolio returns
        portfolio_returns = np.zeros((n_simulations, days_forward))
        
        for asset, weight in allocations.items():
            if weight > 0:
                asset_returns = simulated_returns[asset]
                portfolio_returns += (weight / 100) * asset_returns
                
        # Calculate portfolio values
        portfolio_values = np.zeros((n_simulations, days_forward + 1))
        portfolio_values[:, 0] = initial_value
        
        for i in range(days_forward):
            portfolio_values[:, i + 1] = portfolio_values[:, i] * (1 + portfolio_returns[:, i])
            
        # Calculate metrics
        final_values = portfolio_values[:, -1]
        total_returns = (final_values / initial_value - 1)
        annual_returns = (final_values / initial_value) ** (365 / days_forward) - 1
        
        # Calculate drawdowns
        drawdowns = []
        for path in portfolio_values:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
            drawdowns.append(abs(max_drawdown))
            
        # Calculate Value at Risk
        var_95 = np.percentile(total_returns * 100, 5)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = annual_returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        
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
            'var_95': abs(var_95),
            'recovery_days': int(avg_recovery_time),
            'simulation_days': days_forward,
            'n_simulations': n_simulations
        }