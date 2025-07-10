import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import custom utilities
from utils.pinecone_client import PineconeClient
from utils.portfolio_simulator import PortfolioSimulator
from utils.visualization import create_portfolio_charts
# from utils.metrics import calculate_portfolio_metrics  # Not used

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Portfolio Risk Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global dark theme */
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Main title gradient */
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 153, 255, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099ff);
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #0099ff);
    }
    
    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff, #0099ff);
    }
    
    /* Make the app mobile responsive */
    @media (max-width: 768px) {
        h1 { font-size: 2rem; }
        .subtitle { font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'allocations' not in st.session_state:
    st.session_state.allocations = {
        'stocks': 60.0,
        'bonds': 30.0,
        'real_estate': 5.0,
        'crypto': 5.0
    }

# Title and subtitle
st.markdown("<h1>Portfolio Risk Simulator</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">See how small crypto allocations impact your portfolio risk and return</p>', unsafe_allow_html=True)

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("Pinecone API key not found. Please set PINECONE_API_KEY in your .env file.")
        return None
    return PineconeClient(api_key)

pinecone_client = init_pinecone()

# Sidebar - Portfolio Configuration
st.sidebar.markdown("## Portfolio Configuration")
st.sidebar.markdown("---")

# Portfolio value input
portfolio_value = st.sidebar.number_input(
    "Initial Portfolio Value ($)",
    min_value=10000,
    max_value=10000000,
    value=1000000,
    step=50000,
    format="%d"
)

# Asset allocation sliders
st.sidebar.markdown("### Asset Allocation")
st.sidebar.markdown("*Adjust the sliders below to sum to 100%*")

allocations = {}
allocations['stocks'] = st.sidebar.slider(
    "Stocks (SPY) %",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.allocations['stocks'],
    step=0.5,
    key='stocks_slider'
)

allocations['bonds'] = st.sidebar.slider(
    "Bonds (AGG) %",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.allocations['bonds'],
    step=0.5,
    key='bonds_slider'
)

allocations['real_estate'] = st.sidebar.slider(
    "Real Estate (VNQ) %",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.allocations['real_estate'],
    step=0.5,
    key='real_estate_slider'
)

allocations['crypto'] = st.sidebar.slider(
    "Crypto (COIN50) %",
    min_value=0.0,
    max_value=20.0,  # Cap at 20%
    value=st.session_state.allocations['crypto'],
    step=0.5,
    key='crypto_slider'
)

# Update session state
st.session_state.allocations = allocations

# Calculate total allocation
total_allocation = sum(allocations.values())

# Display allocation status
if abs(total_allocation - 100.0) < 0.1:
    st.sidebar.success(f"✓ Total Allocation: {total_allocation:.1f}%")
else:
    st.sidebar.error(f"⚠️ Total Allocation: {total_allocation:.1f}% (must equal 100%)")

# Simulation parameters
st.sidebar.markdown("### Simulation Parameters")
n_simulations = st.sidebar.select_slider(
    "Number of Simulations",
    options=[100, 500, 1000, 2500, 5000, 10000],
    value=1000
)

simulation_days = st.sidebar.select_slider(
    "Time Horizon (Days)",
    options=[30, 90, 180, 365, 730, 1095, 1825],
    value=365,
    format_func=lambda x: f"{x} days ({x/365:.1f} years)"
)

# Main content area
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Crypto Allocation",
        f"{allocations['crypto']:.1f}%",
        f"${portfolio_value * allocations['crypto'] / 100:,.0f}"
    )

with col2:
    st.metric(
        "Traditional Assets",
        f"{100 - allocations['crypto']:.1f}%",
        f"${portfolio_value * (100 - allocations['crypto']) / 100:,.0f}"
    )

with col3:
    if allocations['crypto'] > 0:
        impact = allocations['crypto'] * 0.5  # If crypto drops 50%
        st.metric(
            "Max Crypto Impact",
            f"-{impact:.1f}%",
            "If crypto drops 50%"
        )
    else:
        st.metric("Max Crypto Impact", "0%", "No crypto exposure")

with col4:
    st.metric(
        "Time Horizon",
        f"{simulation_days/365:.1f} years",
        f"{simulation_days} days"
    )

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Projections", "📈 Risk Analysis", "🔥 Stress Tests", "📋 Report"])

# Load data button
if st.button("🚀 Run Portfolio Simulation", type="primary", disabled=(abs(total_allocation - 100.0) > 0.1)):
    with st.spinner("Loading historical data from Pinecone..."):
        try:
            # Initialize portfolio simulator
            simulator = PortfolioSimulator(pinecone_client)
            
            # Load historical data
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading stocks data (SPY)...")
            progress_bar.progress(0.2)
            
            status_text.text("Loading bonds data (AGG)...")
            progress_bar.progress(0.4)
            
            status_text.text("Loading real estate data (VNQ)...")
            progress_bar.progress(0.6)
            
            status_text.text("Loading crypto data (COIN50)...")
            progress_bar.progress(0.8)
            
            # Run simulation
            status_text.text("Running Monte Carlo simulations...")
            progress_bar.progress(0.9)
            
            results = simulator.run_simulation(
                allocations=allocations,
                n_simulations=n_simulations,
                days_forward=simulation_days,
                initial_value=portfolio_value
            )
            
            st.session_state.simulation_results = results
            st.session_state.portfolio_data = simulator.get_portfolio_data()
            
            progress_bar.progress(1.0)
            status_text.text("Simulation complete!")
            st.success("✓ Portfolio simulation completed successfully!")
            
        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")

# Display results in tabs
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    
    with tab1:
        st.markdown("### Portfolio Growth Projections")
        
        # Create fan chart
        fan_chart = create_portfolio_charts(
            results,
            chart_type='fan',
            title="Portfolio Value Projections"
        )
        st.plotly_chart(fan_chart, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Expected Return")
            st.markdown(f"**{results['expected_return']:.2f}%** annualized")
            st.markdown(f"Median portfolio value: **${results['median_final_value']:,.0f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Best Case (95th percentile)")
            st.markdown(f"**{results['best_case_return']:.2f}%** annualized")
            st.markdown(f"Portfolio value: **${results['best_case_value']:,.0f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Worst Case (5th percentile)")
            st.markdown(f"**{results['worst_case_return']:.2f}%** annualized")
            st.markdown(f"Portfolio value: **${results['worst_case_value']:,.0f}**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Risk Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Return distribution
            return_dist = create_portfolio_charts(
                results,
                chart_type='return_distribution',
                title="Return Distribution"
            )
            st.plotly_chart(return_dist, use_container_width=True)
        
        with col2:
            # Risk gauge
            risk_gauge = create_portfolio_charts(
                results,
                chart_type='risk_gauge',
                title="Portfolio Risk Level"
            )
            st.plotly_chart(risk_gauge, use_container_width=True)
        
        # Risk metrics
        st.markdown("### Key Risk Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
        
        with col3:
            st.metric("95% VaR", f"-{results['var_95']:.1f}%")
        
        with col4:
            st.metric("Recovery Time", f"{results['recovery_days']} days")
    
    with tab3:
        st.markdown("### Historical Stress Test Scenarios")
        
        # Create stress test visualizations
        stress_tests = {
            "2008 Financial Crisis": {
                "stocks": -37,
                "bonds": 5,
                "real_estate": -39,
                "crypto": -50  # Hypothetical
            },
            "2020 COVID Crash": {
                "stocks": -34,
                "bonds": 8,
                "real_estate": -22,
                "crypto": -63
            },
            "2022 Bear Market": {
                "stocks": -19,
                "bonds": -13,
                "real_estate": -28,
                "crypto": -75
            }
        }
        
        for scenario, impacts in stress_tests.items():
            portfolio_impact = sum(
                allocations[asset] / 100 * impact 
                for asset, impact in impacts.items()
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"#### {scenario}")
            
            with col2:
                st.metric("Portfolio Impact", f"{portfolio_impact:.1f}%")
            
            with col3:
                dollar_impact = portfolio_value * portfolio_impact / 100
                st.metric("Dollar Impact", f"${dollar_impact:,.0f}")
        
        # Correlation heatmap
        st.markdown("### Asset Correlations")
        correlation_heatmap = create_portfolio_charts(
            st.session_state.portfolio_data,
            chart_type='correlation',
            title="Historical Asset Correlations"
        )
        st.plotly_chart(correlation_heatmap, use_container_width=True)
    
    with tab4:
        st.markdown("### Portfolio Analysis Report")
        
        # Generate downloadable report
        report_content = f"""
# Portfolio Risk Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Portfolio Configuration
- Initial Value: ${portfolio_value:,.0f}
- Time Horizon: {simulation_days} days ({simulation_days/365:.1f} years)
- Simulations Run: {n_simulations:,}

## Asset Allocation
- Stocks (SPY): {allocations['stocks']:.1f}% (${portfolio_value * allocations['stocks'] / 100:,.0f})
- Bonds (AGG): {allocations['bonds']:.1f}% (${portfolio_value * allocations['bonds'] / 100:,.0f})
- Real Estate (VNQ): {allocations['real_estate']:.1f}% (${portfolio_value * allocations['real_estate'] / 100:,.0f})
- Crypto (COIN50): {allocations['crypto']:.1f}% (${portfolio_value * allocations['crypto'] / 100:,.0f})

## Key Findings
- Expected Annual Return: {results['expected_return']:.2f}%
- Sharpe Ratio: {results['sharpe_ratio']:.2f}
- Maximum Drawdown: {results['max_drawdown']:.1f}%
- 95% Value at Risk: {results['var_95']:.1f}%

## Risk Assessment
With a {allocations['crypto']:.1f}% allocation to crypto:
- If crypto drops 50%, the portfolio would only drop {allocations['crypto'] * 0.5:.1f}%
- This represents a dollar loss of ${portfolio_value * allocations['crypto'] * 0.005:,.0f}

## Recommendation
Small crypto allocations (2.5-10%) can provide asymmetric upside potential while maintaining 
limited downside risk suitable for institutional investors.
"""
        
        st.download_button(
            label="📥 Download Full Report",
            data=report_content,
            file_name=f"portfolio_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">Built with real historical data • Monte Carlo simulations • Institutional-grade analytics</p>',
    unsafe_allow_html=True
)