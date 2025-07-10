"""
Visualization utilities for portfolio simulations
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, Optional

def create_portfolio_charts(data: Dict, chart_type: str, title: str = "") -> go.Figure:
    """
    Create various portfolio visualization charts
    
    Args:
        data: Simulation results or portfolio data
        chart_type: Type of chart ('fan', 'return_distribution', 'risk_gauge', 'correlation')
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
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=['Stocks', 'Bonds', 'Real Estate', 'Crypto'],
        y=['Stocks', 'Bonds', 'Real Estate', 'Crypto'],
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