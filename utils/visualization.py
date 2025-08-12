"""
Visualization utilities for portfolio simulations
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Optional, List


def create_portfolio_charts(data: Dict, chart_type: str, title: str = "") -> go.Figure:
    """
    Create various portfolio visualization charts
    
    Args:
        data: Simulation results or portfolio data
        chart_type: Type of chart ('fan', 'return_distribution', 'risk_gauge', 'correlation',
                   'protection_participation', 'time_horizon', 'regret_minimization', 'break_even')
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    
    if chart_type == 'fan':
        return create_fan_chart(data, title)
    elif chart_type == 'return_distribution':
        return create_return_distribution(data, title)
    elif chart_type == 'risk_gauge':
        return create_risk_gauge(data, title)
    elif chart_type == 'correlation':
        return create_correlation_heatmap(data, title)
    elif chart_type == 'protection_participation':
        return create_protection_participation_chart(data, title)
    elif chart_type == 'time_horizon':
        return create_time_horizon_chart(data, title)
    elif chart_type == 'regret_minimization':
        return create_regret_chart(data, title)
    elif chart_type == 'break_even':
        return create_break_even_chart(data, title)
    elif chart_type == 'comparison':
        return create_comparison_chart(data, title)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")


def create_fan_chart(results: Dict, title: str) -> go.Figure:
    """Create fan chart showing portfolio projections"""
    
    portfolio_values = results['portfolio_values']
    simulation_days = results['simulation_days']
    
    # Calculate percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = {}
    
    for p in percentiles:
        percentile_values[p] = np.percentile(portfolio_values, p, axis=0)
    
    # Create time axis
    days = np.arange(simulation_days + 1)
    
    fig = go.Figure()
    
    # Add percentile bands
    # 5-95% band (lightest)
    fig.add_trace(go.Scatter(
        x=days,
        y=percentile_values[95],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0, 212, 255, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=percentile_values[5],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0, 212, 255, 0)', width=0),
        name='5-95th percentile',
        fillcolor='rgba(0, 212, 255, 0.1)',
        hoverinfo='skip'
    ))
    
    # 10-90% band
    fig.add_trace(go.Scatter(
        x=days,
        y=percentile_values[90],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0, 212, 255, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=percentile_values[10],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0, 212, 255, 0)', width=0),
        name='10-90th percentile',
        fillcolor='rgba(0, 212, 255, 0.15)',
        hoverinfo='skip'
    ))
    
    # 25-75% band (darkest)
    fig.add_trace(go.Scatter(
        x=days,
        y=percentile_values[75],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0, 212, 255, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=days,
        y=percentile_values[25],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0, 212, 255, 0)', width=0),
        name='25-75th percentile',
        fillcolor='rgba(0, 212, 255, 0.2)',
        hoverinfo='skip'
    ))
    
    # Add median line
    fig.add_trace(go.Scatter(
        x=days,
        y=percentile_values[50],
        mode='lines',
        line=dict(color='#00d4ff', width=3),
        name='Median projection',
        hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Days Forward",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)',
            tickformat='$,.0f'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_return_distribution(results: Dict, title: str) -> go.Figure:
    """Create histogram of returns"""
    
    annual_returns = results['annual_returns'] * 100  # Convert to percentage
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=annual_returns,
        nbinsx=50,
        name='Return Distribution',
        marker=dict(
            color='rgba(0, 212, 255, 0.6)',
            line=dict(color='#00d4ff', width=1)
        ),
        hovertemplate='Return: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add vertical lines for key percentiles
    median_return = np.median(annual_returns)
    p5_return = np.percentile(annual_returns, 5)
    p95_return = np.percentile(annual_returns, 95)
    
    fig.add_vline(x=median_return, line_dash="dash", line_color="#00ff00",
                  annotation_text=f"Median: {median_return:.1f}%")
    fig.add_vline(x=p5_return, line_dash="dash", line_color="#ff0000",
                  annotation_text=f"5th %ile: {p5_return:.1f}%")
    fig.add_vline(x=p95_return, line_dash="dash", line_color="#00ff00",
                  annotation_text=f"95th %ile: {p95_return:.1f}%")
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Annual Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        )
    )
    
    return fig


def create_risk_gauge(results: Dict, title: str) -> go.Figure:
    """Create risk gauge visualization"""
    
    # Calculate risk score based on multiple factors
    volatility = np.std(results['annual_returns']) * 100
    max_drawdown = results['max_drawdown']
    var_95 = results['var_95']
    
    # Normalize to 0-100 scale
    risk_score = min(100, (volatility + max_drawdown + var_95) / 3)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00d4ff"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    # Add risk level text
    if risk_score < 25:
        risk_level = "Low Risk"
    elif risk_score < 50:
        risk_level = "Moderate Risk"
    elif risk_score < 75:
        risk_level = "High Risk"
    else:
        risk_level = "Very High Risk"
        
    fig.add_annotation(
        x=0.5, y=0.3,
        text=risk_level,
        showarrow=False,
        font=dict(size=20, color="white")
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    return fig


def create_correlation_heatmap(portfolio_data: Dict, title: str) -> go.Figure:
    """Create correlation heatmap for assets"""
    
    if 'correlation_matrix' not in portfolio_data:
        return go.Figure()
    
    corr_matrix = portfolio_data['correlation_matrix']
    
    # Ensure consistent order ['stocks','bonds','real_estate','crypto'] where available
    desired_order = [col for col in ['stocks', 'bonds', 'real_estate', 'crypto'] if col in corr_matrix.columns]
    if desired_order:
        corr_matrix = corr_matrix.reindex(index=desired_order, columns=desired_order)
        axis_labels = [label.title().replace('_', ' ') for label in desired_order]
    else:
        axis_labels = list(corr_matrix.columns)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=axis_labels,
        y=axis_labels,
        colorscale=[
            [0, '#ff0000'],
            [0.5, '#ffffff'],
            [1, '#00d4ff']
        ],
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(side='bottom'),
        width=600,
        height=500
    )
    
    return fig

def create_protection_participation_chart(data: Dict, title: str) -> go.Figure:
    """Create the killer visualization: Protection vs Participation"""
    
    crypto_allocations = data['allocations']
    downside_protection = data['downside_protection']
    upside_participation = data['upside_participation']
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=downside_protection,
        y=upside_participation,
        mode='markers+text',
        marker=dict(
            size=[10 + alloc * 2 for alloc in crypto_allocations],
            color=crypto_allocations,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Crypto %"),
            line=dict(color='white', width=2)
        ),
        text=[f"{alloc}%" for alloc in crypto_allocations],
        textposition="top center",
        hovertemplate=(
            'Crypto Allocation: %{text}<br>' +
            'Downside Protection: %{x:.2f}%<br>' +
            'Upside Participation: %{y:.2f}%<br>' +
            '<extra></extra>'
        )
    ))
    
    # Add sweet spot annotation
    sweet_spot_idx = data.get('sweet_spot_idx', 2)  # Default to 5% allocation
    fig.add_annotation(
        x=downside_protection[sweet_spot_idx],
        y=upside_participation[sweet_spot_idx],
        text="Sweet Spot",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#00ff00",
        ax=40,
        ay=-40,
        font=dict(size=14, color="#00ff00")
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Downside Protection (Less Loss in Bad Scenarios) →",
        yaxis_title="Upside Participation (More Gain in Good Scenarios) →",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.3)',
            tickformat='.1f',
            ticksuffix='%'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.3)',
            tickformat='.1f',
            ticksuffix='%'
        ),
        height=600
    )
    
    return fig

def create_time_horizon_chart(data: Dict, title: str) -> go.Figure:
    """Create time horizon analysis chart"""
    
    fig = go.Figure()
    
    # Add traces for each time horizon
    for horizon_name, horizon_data in data.items():
        crypto_allocations, sharpe_ratios = zip(*horizon_data['sharpe_ratios'])
        
        fig.add_trace(go.Scatter(
            x=crypto_allocations,
            y=sharpe_ratios,
            mode='lines+markers',
            name=horizon_name,
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate=(
                f'{horizon_name}<br>' +
                'Crypto: %{x}%<br>' +
                'Sharpe Ratio: %{y:.3f}<br>' +
                '<extra></extra>'
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Crypto Allocation (%)",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        height=500
    )
    
    # Add annotation about longer horizons
    fig.add_annotation(
        x=0.99,
        y=0.01,
        xref="paper",
        yref="paper",
        text="Longer time horizons favor higher crypto allocations",
        showarrow=False,
        font=dict(size=12, color="#00d4ff"),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor="#00d4ff",
        borderwidth=1
    )
    
    return fig

def create_regret_chart(data: Dict, title: str) -> go.Figure:
    """Create regret minimization heatmap"""
    
    allocations = data['allocations']
    regret_matrix = data['regret_matrix']
    
    # Create matrix for heatmap
    z_values = []
    for i, item in enumerate(regret_matrix):
        row = []
        for j, alloc in enumerate(allocations):
            if i == j:
                row.append(0)  # No regret comparing to self
            else:
                row.append(item['regret_vs_others'].get(alloc, 0) * 100)
        z_values.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=[f"{a}%" for a in allocations],
        y=[f"{a}%" for a in allocations],
        colorscale=[
            [0, '#00ff00'],  # Green for low regret
            [0.5, '#ffff00'],  # Yellow for medium regret
            [1, '#ff0000']  # Red for high regret
        ],
        text=np.round(z_values, 1),
        texttemplate='%{text}%',
        textfont={"size": 12},
        hovertemplate=(
            'Your Allocation: %{y}<br>' +
            'Alternative: %{x}<br>' +
            'Regret Probability: %{z:.1f}%<br>' +
            '<extra></extra>'
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Alternative Allocation",
        yaxis_title="Your Allocation",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left'),
        height=500
    )
    
    # Add annotation for minimum regret
    min_regret_idx = np.argmin([sum(row) for row in z_values])
    fig.add_annotation(
        x=min_regret_idx,
        y=min_regret_idx,
        text="Minimum Regret",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#00ff00",
        font=dict(size=12, color="#00ff00")
    )
    
    return fig

def create_break_even_chart(data: Dict, title: str) -> go.Figure:
    """Create break-even analysis visualization"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Crypto Gain Needed to Offset Losses", 
                       "Crypto Drop Allowed Before Losing Gains"),
        horizontal_spacing=0.15
    )
    
    crypto_allocations = data['crypto_allocations']
    gains_needed = data['gains_needed']
    drops_allowed = data['drops_allowed']
    
    # Left chart - Gains needed
    fig.add_trace(
        go.Bar(
            x=crypto_allocations,
            y=gains_needed,
            marker_color='#ff6b6b',
            name='Gain Needed',
            hovertemplate='Crypto: %{x}%<br>Gain Needed: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Right chart - Drops allowed
    fig.add_trace(
        go.Bar(
            x=crypto_allocations,
            y=drops_allowed,
            marker_color='#51cf66',
            name='Drop Allowed',
            hovertemplate='Crypto: %{x}%<br>Drop Allowed: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Crypto Allocation (%)", row=1, col=1)
    fig.update_xaxes(title_text="Crypto Allocation (%)", row=1, col=2)
    fig.update_yaxes(title_text="Required Gain (%)", row=1, col=1)
    fig.update_yaxes(title_text="Allowed Drop (%)", row=1, col=2)
    
    fig.update_layout(
        title_text=title,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        height=400
    )
    
    return fig

def create_comparison_chart(data: Dict, title: str) -> go.Figure:
    """Create side-by-side portfolio comparison"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Expected Returns", "Risk Metrics", 
                       "Portfolio Growth", "Return Distribution"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "histogram"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    with_crypto = data['with_crypto']
    without_crypto = data['without_crypto']
    
    # Expected returns comparison
    fig.add_trace(
        go.Bar(
            x=['Without Crypto', 'With Crypto'],
            y=[without_crypto['expected_return'], with_crypto['expected_return']],
            marker_color=['#666666', '#00d4ff'],
            text=[f"{without_crypto['expected_return']:.2f}%", 
                  f"{with_crypto['expected_return']:.2f}%"],
            textposition='auto',
            hovertemplate='%{x}<br>Return: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Risk metrics comparison
    metrics = ['Max Drawdown', 'VaR 95%', 'Sharpe Ratio']
    without_values = [without_crypto['max_drawdown'], 
                     without_crypto['var_95'],
                     without_crypto['sharpe_ratio']]
    with_values = [with_crypto['max_drawdown'], 
                   with_crypto['var_95'],
                   with_crypto['sharpe_ratio']]
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=without_values,
            name='Without Crypto',
            marker_color='#666666',
            hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=with_values,
            name='With Crypto',
            marker_color='#00d4ff',
            hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Portfolio growth paths
    days = np.arange(with_crypto['simulation_days'] + 1)
    
    # Calculate median paths
    without_median = np.median(without_crypto['portfolio_values'], axis=0)
    with_median = np.median(with_crypto['portfolio_values'], axis=0)
    
    fig.add_trace(
        go.Scatter(
            x=days,
            y=without_median,
            mode='lines',
            name='Without Crypto',
            line=dict(color='#666666', width=2),
            hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=days,
            y=with_median,
            mode='lines',
            name='With Crypto',
            line=dict(color='#00d4ff', width=2),
            hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Return distributions
    fig.add_trace(
        go.Histogram(
            x=without_crypto['annual_returns'] * 100,
            name='Without Crypto',
            marker_color='#666666',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability',
            hovertemplate='Return: %{x:.1f}%<br>Probability: %{y:.1%}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Histogram(
            x=with_crypto['annual_returns'] * 100,
            name='With Crypto',
            marker_color='#00d4ff',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability',
            hovertemplate='Return: %{x:.1f}%<br>Probability: %{y:.1%}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_yaxes(title_text="Annual Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1, tickformat="$,.0f")
    fig.update_xaxes(title_text="Annual Return (%)", row=2, col=2)
    fig.update_yaxes(title_text="Probability", row=2, col=2, tickformat=".1%")
    
    # Update layout
    fig.update_layout(
        title_text=title,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig