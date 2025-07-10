"""
Advanced analytics for portfolio risk assessment
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) - expected loss beyond VaR
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaR as a positive percentage
    """
    # Calculate VaR threshold
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    
    # Calculate mean of returns worse than VaR
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return 0
    
    cvar = -np.mean(tail_returns) * 100  # Return as positive percentage
    return cvar

def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0) -> float:
    """
    Calculate Omega ratio - probability-weighted ratio of gains vs losses
    
    Args:
        returns: Array of returns
        threshold: Minimum acceptable return (default 0)
        
    Returns:
        Omega ratio
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    
    if losses.sum() == 0:
        return np.inf
    
    return gains.sum() / losses.sum()

def time_horizon_analysis(simulator, allocations: Dict[str, float], 
                         initial_value: float, n_simulations: int = 1000) -> Dict:
    """
    Analyze optimal crypto allocation for different time horizons
    
    Returns:
        Dictionary with results for each time horizon
    """
    horizons = {
        '1 Year': 365,
        '3 Years': 1095,
        '5 Years': 1825,
        '10 Years': 3650
    }
    
    results = {}
    
    for horizon_name, days in horizons.items():
        # Test different crypto allocations
        crypto_allocations = [0, 2.5, 5, 7.5, 10, 15, 20]
        sharpe_ratios = []
        
        for crypto_pct in crypto_allocations:
            # Adjust allocations
            test_allocations = allocations.copy()
            
            # Scale down other allocations proportionally
            scale_factor = (100 - crypto_pct) / (100 - allocations.get('crypto', 0))
            for asset in test_allocations:
                if asset != 'crypto':
                    test_allocations[asset] *= scale_factor
            test_allocations['crypto'] = crypto_pct
            
            # Run simulation
            sim_results = simulator.run_simulation(
                allocations=test_allocations,
                n_simulations=n_simulations,
                days_forward=days,
                initial_value=initial_value
            )
            
            sharpe_ratios.append(sim_results['sharpe_ratio'])
        
        # Find optimal allocation
        optimal_idx = np.argmax(sharpe_ratios)
        optimal_crypto = crypto_allocations[optimal_idx]
        
        results[horizon_name] = {
            'optimal_crypto_allocation': optimal_crypto,
            'sharpe_ratios': list(zip(crypto_allocations, sharpe_ratios)),
            'days': days
        }
    
    return results

def calculate_regret_matrix(simulator, initial_value: float, 
                          n_simulations: int = 1000, days_forward: int = 365) -> Dict:
    """
    Calculate regret matrix for different crypto allocations
    
    Returns:
        Dictionary with regret probabilities
    """
    crypto_allocations = [0, 2.5, 5, 7.5, 10]
    
    # Base allocations (will be adjusted)
    base_allocations = {
        'stocks': 60,
        'bonds': 30,
        'real_estate': 10,
        'crypto': 0
    }
    
    results = []
    
    for crypto_pct in crypto_allocations:
        # Adjust allocations
        test_allocations = base_allocations.copy()
        scale_factor = (100 - crypto_pct) / 100
        test_allocations['stocks'] = base_allocations['stocks'] * scale_factor
        test_allocations['bonds'] = base_allocations['bonds'] * scale_factor
        test_allocations['real_estate'] = base_allocations['real_estate'] * scale_factor
        test_allocations['crypto'] = crypto_pct
        
        # Run simulation
        sim_results = simulator.run_simulation(
            allocations=test_allocations,
            n_simulations=n_simulations,
            days_forward=days_forward,
            initial_value=initial_value
        )
        
        results.append({
            'crypto_allocation': crypto_pct,
            'final_values': sim_results['final_values'],
            'expected_return': sim_results['expected_return'],
            'downside_risk': sim_results['var_95']
        })
    
    # Calculate regret probabilities
    regret_matrix = []
    
    for i, result in enumerate(results):
        regrets = {
            'allocation': result['crypto_allocation'],
            'regret_vs_others': {}
        }
        
        for j, other in enumerate(results):
            if i != j:
                # Probability that this allocation underperforms the other
                underperform_count = np.sum(
                    result['final_values'] < other['final_values']
                )
                regret_prob = underperform_count / n_simulations
                
                regrets['regret_vs_others'][other['crypto_allocation']] = regret_prob
        
        regret_matrix.append(regrets)
    
    return {
        'regret_matrix': regret_matrix,
        'allocations': crypto_allocations,
        'results': results
    }

def calculate_dynamic_correlation(returns_data: Dict[str, pd.Series], 
                                window: int = 60) -> pd.DataFrame:
    """
    Calculate rolling correlations between assets
    
    Args:
        returns_data: Dictionary of return series for each asset
        window: Rolling window size
        
    Returns:
        DataFrame with rolling correlations
    """
    # Create DataFrame from returns
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate rolling correlation for crypto with other assets
    rolling_corr = {}
    
    if 'crypto' in returns_df.columns:
        for asset in returns_df.columns:
            if asset != 'crypto':
                rolling_corr[f'crypto_vs_{asset}'] = (
                    returns_df['crypto'].rolling(window)
                    .corr(returns_df[asset])
                )
    
    return pd.DataFrame(rolling_corr)

def stress_test_correlation_analysis(returns_data: Dict[str, pd.Series],
                                   stress_threshold: float = -0.05) -> Dict:
    """
    Analyze how correlations change during market stress
    
    Args:
        returns_data: Dictionary of return series
        stress_threshold: Return threshold for stress periods
        
    Returns:
        Correlation comparison between normal and stress periods
    """
    returns_df = pd.DataFrame(returns_data)
    
    # Identify stress periods (when stock market drops significantly)
    stress_periods = returns_df['stocks'] < stress_threshold
    normal_periods = ~stress_periods
    
    # Calculate correlations for each period
    stress_corr = returns_df[stress_periods].corr()
    normal_corr = returns_df[normal_periods].corr()
    
    # Calculate correlation increase during stress
    corr_change = stress_corr - normal_corr
    
    return {
        'normal_correlation': normal_corr,
        'stress_correlation': stress_corr,
        'correlation_change': corr_change,
        'stress_periods_pct': stress_periods.sum() / len(stress_periods) * 100
    }