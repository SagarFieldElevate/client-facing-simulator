"""
Portfolio metrics calculation utilities
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def calculate_portfolio_metrics(portfolio_values: np.ndarray, 
                              initial_value: float,
                              time_horizon_days: int) -> Dict:
    """
    Calculate comprehensive portfolio metrics
    
    Args:
        portfolio_values: Array of portfolio values (simulations x days)
        initial_value: Initial portfolio value
        time_horizon_days: Number of days in simulation
        
    Returns:
        Dictionary of calculated metrics
    """
    
    # Final values and returns
    final_values = portfolio_values[:, -1]
    total_returns = (final_values / initial_value - 1)
    annual_returns = (final_values / initial_value) ** (365 / time_horizon_days) - 1
    
    # Basic statistics
    metrics = {
        'mean_return': np.mean(annual_returns) * 100,
        'median_return': np.median(annual_returns) * 100,
        'std_return': np.std(annual_returns) * 100,
        'skewness': calculate_skewness(annual_returns),
        'kurtosis': calculate_kurtosis(annual_returns)
    }
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        metrics[f'percentile_{p}'] = np.percentile(annual_returns, p) * 100
    
    # Risk metrics
    metrics['var_95'] = calculate_var(total_returns, 0.95) * 100
    metrics['cvar_95'] = calculate_cvar(total_returns, 0.95) * 100
    metrics['max_drawdown'] = calculate_max_drawdown(portfolio_values)
    
    # Performance metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(annual_returns)
    metrics['sortino_ratio'] = calculate_sortino_ratio(annual_returns)
    metrics['calmar_ratio'] = calculate_calmar_ratio(annual_returns, metrics['max_drawdown'])
    metrics['omega_ratio'] = calculate_omega_ratio(annual_returns)
    
    # Win/loss statistics
    metrics['win_rate'] = np.sum(total_returns > 0) / len(total_returns) * 100
    metrics['avg_win'] = np.mean(total_returns[total_returns > 0]) * 100 if np.any(total_returns > 0) else 0
    metrics['avg_loss'] = np.mean(total_returns[total_returns < 0]) * 100 if np.any(total_returns < 0) else 0
    
    return metrics

def calculate_skewness(returns: np.ndarray) -> float:
    """Calculate skewness of returns"""
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    n = len(returns)
    
    if std == 0:
        return 0
        
    skew = n / ((n - 1) * (n - 2)) * np.sum(((returns - mean) / std) ** 3)
    return skew

def calculate_kurtosis(returns: np.ndarray) -> float:
    """Calculate excess kurtosis of returns"""
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    n = len(returns)
    
    if std == 0:
        return 0
        
    kurt = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((returns - mean) / std) ** 4)
    kurt -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    
    return kurt

def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk"""
    return -np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    var = calculate_var(returns, confidence_level)
    return -np.mean(returns[returns <= -var])

def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Calculate maximum drawdown across all simulations"""
    drawdowns = []
    
    for path in portfolio_values:
        running_max = np.maximum.accumulate(path)
        drawdown = (path - running_max) / running_max
        max_dd = np.min(drawdown)
        drawdowns.append(abs(max_dd))
    
    return np.mean(drawdowns) * 100

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0
        
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (uses downside deviation)"""
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
        
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.inf
        
    return np.mean(excess_returns) / downside_std

def calculate_calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
    """Calculate Calmar ratio (return / max drawdown)"""
    if max_drawdown == 0:
        return np.inf
        
    return np.mean(returns) * 100 / max_drawdown

def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0) -> float:
    """Calculate Omega ratio"""
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    
    if losses.sum() == 0:
        return np.inf
    
    return gains.sum() / losses.sum()

def calculate_recovery_time(portfolio_values: np.ndarray) -> float:
    """Calculate average recovery time from drawdowns"""
    recovery_times = []
    
    for path in portfolio_values:
        running_max = np.maximum.accumulate(path)
        in_drawdown = False
        drawdown_start = 0
        
        for i in range(1, len(path)):
            if path[i] < running_max[i] and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif path[i] >= running_max[i] and in_drawdown:
                recovery_times.append(i - drawdown_start)
                in_drawdown = False
    
    return np.mean(recovery_times) if recovery_times else 0

def calculate_stress_test_impact(allocations: Dict[str, float], 
                               stress_scenario: Dict[str, float]) -> float:
    """
    Calculate portfolio impact under stress scenario
    
    Args:
        allocations: Asset allocations (percentages)
        stress_scenario: Asset returns in stress scenario
        
    Returns:
        Total portfolio impact (percentage)
    """
    total_impact = 0
    
    for asset, allocation in allocations.items():
        if asset in stress_scenario:
            impact = allocation / 100 * stress_scenario[asset]
            total_impact += impact
            
    return total_impact

def calculate_loss_framing(portfolio_value: float, loss_percentage: float) -> Dict[str, str]:
    """
    Frame losses in psychologically more acceptable ways
    
    Args:
        portfolio_value: Initial portfolio value
        loss_percentage: Loss as a percentage
        
    Returns:
        Dictionary with different framings
    """
    loss_amount = portfolio_value * loss_percentage / 100
    remaining_value = portfolio_value - loss_amount
    remaining_percentage = 100 - loss_percentage
    
    return {
        'traditional': f"Portfolio drops {loss_percentage:.1f}%",
        'positive_framing': f"You keep {remaining_percentage:.1f}% of your wealth",
        'dollar_remaining': f"${remaining_value:,.0f} protected out of ${portfolio_value:,.0f}",
        'relative_framing': f"Only ${loss_amount:,.0f} at risk in worst case"
    }

def calculate_opportunity_cost(years: int, missed_return: float, 
                              initial_investment: float) -> Dict[str, float]:
    """
    Calculate opportunity cost of not having crypto allocation
    
    Args:
        years: Number of years
        missed_return: Annual return missed (as decimal)
        initial_investment: Initial investment amount
        
    Returns:
        Dictionary with opportunity cost metrics
    """
    final_value_with = initial_investment * (1 + missed_return) ** years
    final_value_without = initial_investment
    
    return {
        'total_missed_gains': final_value_with - final_value_without,
        'percentage_missed': ((final_value_with / final_value_without) - 1) * 100,
        'annual_cost': (final_value_with - final_value_without) / years
    }