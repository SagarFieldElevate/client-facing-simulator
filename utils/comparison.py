"""
Portfolio comparison utilities for with/without crypto analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

def calculate_traditional_allocation(allocations: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate allocation without crypto - crypto allocation goes to stocks
    
    Args:
        allocations: Original allocations including crypto
        
    Returns:
        Allocations with crypto portion added to stocks
    """
    traditional_allocations = allocations.copy()
    crypto_portion = traditional_allocations.pop('crypto', 0)
    
    if crypto_portion == 0:
        return traditional_allocations
    
    # Add crypto allocation to stocks
    traditional_allocations['stocks'] = traditional_allocations.get('stocks', 0) + crypto_portion
    
    return traditional_allocations

def calculate_portfolio_comparison(simulator, allocations: Dict[str, float], 
                                 n_simulations: int, days_forward: int, 
                                 initial_value: float) -> Dict:
    """
    Run portfolio simulation for both with and without crypto
    
    Returns:
        Dictionary containing results for both portfolios
    """
    # Run simulation with crypto
    with_crypto_results = simulator.run_simulation(
        allocations=allocations,
        n_simulations=n_simulations,
        days_forward=days_forward,
        initial_value=initial_value
    )
    
    # Calculate traditional allocation
    traditional_allocations = calculate_traditional_allocation(allocations)
    
    # Run simulation without crypto
    without_crypto_results = simulator.run_simulation(
        allocations=traditional_allocations,
        n_simulations=n_simulations,
        days_forward=days_forward,
        initial_value=initial_value
    )
    
    # Calculate incremental metrics
    incremental_return = (with_crypto_results['expected_return'] - 
                         without_crypto_results['expected_return'])
    incremental_risk = (with_crypto_results['max_drawdown'] - 
                       without_crypto_results['max_drawdown'])
    
    return {
        'with_crypto': with_crypto_results,
        'without_crypto': without_crypto_results,
        'incremental_return': incremental_return,
        'incremental_risk': incremental_risk,
        'allocations': {
            'with_crypto': allocations,
            'without_crypto': traditional_allocations
        }
    }

def calculate_protection_participation(comparison_results: Dict) -> Tuple[List[float], List[float]]:
    """
    Calculate protection vs participation metrics for different crypto allocations
    
    Returns:
        Tuple of (downside_protection, upside_participation) lists
    """
    # Extract portfolio values
    with_values = comparison_results['with_crypto']['portfolio_values']
    without_values = comparison_results['without_crypto']['portfolio_values']
    
    # Calculate downside protection (how much less you lose in bad scenarios)
    p5_with = np.percentile(with_values[:, -1], 5)
    p5_without = np.percentile(without_values[:, -1], 5)
    downside_protection = (p5_with - p5_without) / p5_without * 100
    
    # Calculate upside participation (how much more you gain in good scenarios)
    p95_with = np.percentile(with_values[:, -1], 95)
    p95_without = np.percentile(without_values[:, -1], 95)
    upside_participation = (p95_with - p95_without) / p95_without * 100
    
    return downside_protection, upside_participation

def calculate_break_even_scenarios(allocations: Dict[str, float], 
                                 asset_returns: Dict[str, float]) -> Dict:
    """
    Calculate break-even scenarios for crypto allocation
    
    Args:
        allocations: Portfolio allocations
        asset_returns: Expected returns for each asset
        
    Returns:
        Dictionary with break-even calculations
    """
    crypto_alloc = allocations.get('crypto', 0) / 100
    
    if crypto_alloc == 0:
        return {'crypto_needed': np.inf, 'traditional_loss_covered': 0}
    
    # Calculate weighted traditional return
    traditional_return = 0
    total_traditional = 0
    for asset, alloc in allocations.items():
        if asset != 'crypto':
            traditional_return += (alloc / 100) * asset_returns.get(asset, 0)
            total_traditional += alloc / 100
    
    # How much crypto needs to rise to offset traditional losses
    if traditional_return < 0:
        crypto_gain_needed = abs(traditional_return * total_traditional / crypto_alloc)
    else:
        crypto_gain_needed = 0
    
    # How much crypto can drop before wiping out traditional gains
    if traditional_return > 0:
        crypto_drop_allowed = traditional_return * total_traditional / crypto_alloc
    else:
        crypto_drop_allowed = 0
    
    return {
        'crypto_gain_needed': crypto_gain_needed * 100,  # Convert to percentage
        'crypto_drop_allowed': crypto_drop_allowed * 100,
        'traditional_return': traditional_return * 100
    }