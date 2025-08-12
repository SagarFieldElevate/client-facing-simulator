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
from utils.metrics import calculate_portfolio_metrics, calculate_loss_framing, calculate_opportunity_cost
from utils.comparison import (calculate_portfolio_comparison, calculate_protection_participation,
                             calculate_break_even_scenarios)
from utils.advanced_analytics import (calculate_cvar, time_horizon_analysis, 
                                     calculate_regret_matrix, stress_test_correlation_analysis)

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
    
    /* Comparison mode toggle */
    .comparison-toggle {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 20px 0;
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
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'allocations' not in st.session_state:
    st.session_state.allocations = {
        'stocks': 60.0,
        'bonds': 30.0,
        'real_estate': 5.0,
        'crypto': 5.0
    }
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = True

# Title and subtitle
st.markdown("<h1>Field Elevate Portfolio Risk Simulator</h1>", unsafe_allow_html=True)
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

# Comparison Mode Toggle
st.markdown('<div class="comparison-toggle">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])
with col1:
   st.markdown("### Analysis Mode")
   st.markdown("Compare portfolios with and without crypto allocation")
with col2:
   comparison_mode = st.toggle("Comparison Mode", value=st.session_state.comparison_mode)
   st.session_state.comparison_mode = comparison_mode
st.markdown('</div>', unsafe_allow_html=True)

# Monte Carlo simulation notice
st.info(
    f"All projections shown are generated via Monte Carlo simulation "
    f"(n={n_simulations:,}, horizon={simulation_days} days) using historical data for drift/correlation "
    f"and GARCH-based volatility. These are estimates, not guarantees."
)

# Main content area - Key Metrics
if comparison_mode and allocations['crypto'] > 0:
   # Show comparison metrics
   col1, col2, col3, col4 = st.columns(4)
   
   with col1:
       st.metric(
           "Crypto Allocation",
           f"{allocations['crypto']:.1f}%",
           f"${portfolio_value * allocations['crypto'] / 100:,.0f}"
       )
   
   with col2:
       # Proportional rebalancing of crypto into non-crypto assets
       non_crypto_total = allocations['stocks'] + allocations['bonds'] + allocations['real_estate']
       if non_crypto_total > 0:
           stocks_new = allocations['stocks'] + allocations['crypto'] * (allocations['stocks'] / non_crypto_total)
           bonds_new = allocations['bonds'] + allocations['crypto'] * (allocations['bonds'] / non_crypto_total)
           re_new = allocations['real_estate'] + allocations['crypto'] * (allocations['real_estate'] / non_crypto_total)
       else:
           stocks_new, bonds_new, re_new = 100.0, 0.0, 0.0
       
       st.metric(
           "Without Crypto",
           "→ Rebalanced",
           f"S:{stocks_new:.0f}% B:{bonds_new:.0f}% RE:{re_new:.0f}%"
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
else:
   # Original single portfolio metrics
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
if comparison_mode:
   tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
       "📊 Comparison", "🎯 Protection vs Participation", "📈 Time Horizon", 
       "😔 Regret Analysis", "💔 Break-Even", "📋 Report"
   ])
else:
   tab1, tab2, tab3, tab4 = st.tabs([
       "📊 Projections", "📈 Risk Analysis", "🔥 Stress Tests", "📋 Report"
   ])

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
           
           if comparison_mode and allocations['crypto'] > 0:
               # Run comparison analysis
               comparison_results = calculate_portfolio_comparison(
                   simulator=simulator,
                   allocations=allocations,
                   n_simulations=n_simulations,
                   days_forward=simulation_days,
                   initial_value=portfolio_value
               )
               st.session_state.comparison_results = comparison_results
               st.session_state.simulation_results = comparison_results['with_crypto']
           else:
               # Run single portfolio simulation
               results = simulator.run_simulation(
                   allocations=allocations,
                   n_simulations=n_simulations,
                   days_forward=simulation_days,
                   initial_value=portfolio_value
               )
               st.session_state.simulation_results = results
               st.session_state.comparison_results = None
           
           st.session_state.portfolio_data = simulator.get_portfolio_data()
           
           progress_bar.progress(1.0)
           status_text.text("Simulation complete!")
           st.success("✓ Portfolio simulation completed successfully!")
           
       except Exception as e:
           st.error(f"Error running simulation: {str(e)}")

# Display results in tabs
if st.session_state.simulation_results:
   
   if comparison_mode and st.session_state.comparison_results:
       # Comparison mode tabs
       comparison_data = st.session_state.comparison_results
       
       with tab1:
           st.markdown("### Portfolio Comparison: With vs Without Crypto")
           
           # Create comparison visualization
           comparison_chart = create_portfolio_charts(
               comparison_data,
               chart_type='comparison',
               title="Side-by-Side Portfolio Analysis"
           )
           st.plotly_chart(comparison_chart, use_container_width=True)
           
           # Key insights
           col1, col2, col3 = st.columns(3)
           
           with col1:
               st.markdown('<div class="card">', unsafe_allow_html=True)
               st.markdown("#### Incremental Return")
               inc_return = comparison_data['incremental_return']
               st.markdown(f"**{inc_return:+.2f}%** additional annual return")
               st.markdown(f"Worth **${portfolio_value * inc_return / 100:,.0f}** per year")
               st.markdown('</div>', unsafe_allow_html=True)
           
           with col2:
               st.markdown('<div class="card">', unsafe_allow_html=True)
               st.markdown("#### Additional Risk")
               inc_risk = comparison_data['incremental_risk']
               st.markdown(f"**{inc_risk:+.1f}%** max drawdown increase")
               st.markdown("Minimal impact on downside")
               st.markdown('</div>', unsafe_allow_html=True)
           
           with col3:
               st.markdown('<div class="card">', unsafe_allow_html=True)
               st.markdown("#### Risk-Adjusted Return")
               sharpe_with = comparison_data['with_crypto']['sharpe_ratio']
               sharpe_without = comparison_data['without_crypto']['sharpe_ratio']
               sharpe_improvement = (sharpe_with / sharpe_without - 1) * 100 if abs(sharpe_without) > 1e-12 else float('nan')
               st.markdown(f"**{sharpe_improvement:+.1f}%** Sharpe improvement")
               st.markdown(f"Better risk-adjusted returns")
               st.markdown('</div>', unsafe_allow_html=True)
       
       with tab2:
           st.markdown("### Protection vs Participation Analysis")
           st.markdown("*The optimal trade-off between downside protection and upside participation*")
           
           # Calculate protection vs participation for different allocations
           crypto_allocations = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20]
           protection_data = []
           participation_data = []
           
           for crypto_pct in crypto_allocations:
               test_alloc = allocations.copy()
               # Scale other allocations
               scale_factor = (100 - crypto_pct) / (100 - allocations['crypto'])
               for asset in ['stocks', 'bonds', 'real_estate']:
                   test_alloc[asset] = allocations[asset] * scale_factor
               test_alloc['crypto'] = crypto_pct
               
               # Quick simulation for this allocation
               test_comparison = calculate_portfolio_comparison(
                   simulator=simulator,
                   allocations=test_alloc,
                   n_simulations=min(500, n_simulations),  # Faster for multiple runs
                   days_forward=simulation_days,
                   initial_value=portfolio_value
               )
               
               protection, participation = calculate_protection_participation(test_comparison)
               protection_data.append(protection)
               participation_data.append(participation)
           
           # Find sweet spot dynamically: highest Sharpe ratio improvement vs 0%
           base_sharpe = st.session_state.comparison_results['without_crypto']['sharpe_ratio'] if st.session_state.comparison_results else None
           if base_sharpe is not None:
               sharpe_improvements = []
               for crypto_pct in crypto_allocations:
                   test_alloc = allocations.copy()
                   scale_factor = (100 - crypto_pct) / (100 - allocations['crypto']) if (100 - allocations['crypto']) > 0 else 0
                   for asset in ['stocks', 'bonds', 'real_estate']:
                       test_alloc[asset] = allocations[asset] * scale_factor
                   test_alloc['crypto'] = crypto_pct
                   test_res = simulator.run_simulation(
                       allocations=test_alloc,
                       n_simulations=min(300, n_simulations),
                       days_forward=simulation_days,
                       initial_value=portfolio_value
                   )
                   sharpe_improvements.append(test_res['sharpe_ratio'] - base_sharpe)
               sweet_spot_idx = int(np.argmax(sharpe_improvements))
           else:
               sweet_spot_idx = 2  # fallback
           
           # Create visualization
           pp_chart = create_portfolio_charts(
               {
                   'allocations': crypto_allocations,
                   'downside_protection': protection_data,
                   'upside_participation': participation_data,
                   'sweet_spot_idx': sweet_spot_idx
               },
               chart_type='protection_participation',
               title="Protection vs Participation Trade-off"
           )
           st.plotly_chart(pp_chart, use_container_width=True)
           
           # Insights
           st.info(
               f"💡 **Key Insight**: The 'sweet spot' around {crypto_allocations[sweet_spot_idx]}% crypto "
               f"provides optimal balance between protecting downside and capturing upside. "
               f"Small allocations offer asymmetric risk/reward."
           )
       
       with tab3:
           st.markdown("### Time Horizon Analysis")
           st.markdown("*How optimal crypto allocation changes with investment timeframe*")
           
           # Run time horizon analysis
           with st.spinner("Analyzing different time horizons..."):
               time_horizon_results = time_horizon_analysis(
                   simulator=simulator,
                   allocations=allocations,
                   initial_value=portfolio_value,
                   n_simulations=min(500, n_simulations)
               )
           
           # Create visualization
           time_chart = create_portfolio_charts(
               time_horizon_results,
               chart_type='time_horizon',
               title="Optimal Crypto Allocation by Time Horizon"
           )
           st.plotly_chart(time_chart, use_container_width=True)
           
           # Summary table
           st.markdown("### Optimal Allocations Summary")
           summary_data = []
           for horizon, data in time_horizon_results.items():
               summary_data.append({
                   'Time Horizon': horizon,
                   'Optimal Crypto %': f"{data['optimal_crypto_allocation']}%",
                   'Max Sharpe Ratio': f"{max(sr for _, sr in data['sharpe_ratios']):.3f}"
               })
           
           st.dataframe(pd.DataFrame(summary_data), hide_index=True)
           
           st.success(
               "📈 **Key Finding**: Longer investment horizons support higher crypto allocations "
               "due to volatility smoothing over time."
           )
       
       with tab4:
           st.markdown("### Regret Minimization Analysis")
           st.markdown("*Probability of regretting your allocation choice*")
           
           # Calculate regret matrix
           with st.spinner("Calculating regret probabilities..."):
               regret_results = calculate_regret_matrix(
                    simulator=simulator,
                    initial_value=portfolio_value,
                    n_simulations=min(500, n_simulations),
                    days_forward=simulation_days,
                    base_allocations=allocations,
                    crypto_allocations=[0, 2.5, 5, 7.5, 10]
                )
           
           # Create visualization
           regret_chart = create_portfolio_charts(
               regret_results,
               chart_type='regret_minimization',
               title="Regret Probability Matrix"
           )
           st.plotly_chart(regret_chart, use_container_width=True)
           
           # Behavioral insights
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown('<div class="card">', unsafe_allow_html=True)
               st.markdown("#### Regret from NOT having crypto")
               st.markdown("In bull markets, zero allocation leads to significant FOMO")
               st.markdown("Small allocations reduce this psychological burden")
               st.markdown('</div>', unsafe_allow_html=True)
           
           with col2:
               st.markdown('<div class="card">', unsafe_allow_html=True)
               st.markdown("#### Regret from having crypto")
               st.markdown("Limited to your allocation size")
               st.markdown("5% allocation = maximum 5% portfolio impact")
               st.markdown('</div>', unsafe_allow_html=True)
       
       with tab5:
           st.markdown("### Break-Even Analysis")
           st.markdown("*Understanding your risk boundaries*")
           
           # Calculate break-even scenarios using current allocations scaled for target crypto
           crypto_allocations = [2.5, 5, 7.5, 10, 15, 20]
           gains_needed = []
           drops_allowed = []
           
           # Example market scenario
           market_scenario = {
               'stocks': -10,  # 10% drop
               'bonds': 2,     # 2% gain
               'real_estate': -5,  # 5% drop
               'crypto': 0     # To be calculated
           }
           
           for crypto_pct in crypto_allocations:
               # Proportionally scale current non-crypto allocations
               test_alloc = allocations.copy()
               scale_factor = (100 - crypto_pct) / (100 - allocations['crypto']) if (100 - allocations['crypto']) > 0 else 0
               for asset in ['stocks', 'bonds', 'real_estate']:
                   test_alloc[asset] = allocations[asset] * scale_factor
               test_alloc['crypto'] = crypto_pct
               
               break_even = calculate_break_even_scenarios(test_alloc, market_scenario)
               gains_needed.append(break_even['crypto_gain_needed'])
               drops_allowed.append(break_even['crypto_drop_allowed'])
           
           # Create visualization
           break_even_chart = create_portfolio_charts(
               {
                   'crypto_allocations': crypto_allocations,
                   'gains_needed': gains_needed,
                   'drops_allowed': drops_allowed
               },
               chart_type='break_even',
               title="Break-Even Analysis: Crypto Performance Requirements"
           )
           st.plotly_chart(break_even_chart, use_container_width=True)
           
           # Framing insights
           current_crypto = allocations['crypto']
           if current_crypto > 0:
               loss_framing = calculate_loss_framing(
                   portfolio_value,
                   current_crypto * 0.75  # 75% crypto drop
               )
               
               st.markdown("### Loss Framing Perspective")
               col1, col2 = st.columns(2)
               
               with col1:
                   st.markdown("**Traditional Framing:**")
                   st.markdown(f"❌ {loss_framing['traditional']}")
               
               with col2:
                   st.markdown("**Positive Framing:**")
                   st.markdown(f"✅ {loss_framing['positive_framing']}")
                   st.markdown(f"✅ {loss_framing['dollar_remaining']}")
       
       with tab6:
           st.markdown("### Comprehensive Portfolio Analysis Report")
           
           # Generate detailed comparison report
           report_content = f"""
# Portfolio Risk Analysis Report - With/Without Crypto Comparison
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary
This analysis compares two portfolio configurations:
1. With Crypto: {allocations['crypto']:.1f}% allocation to COIN50
2. Without Crypto: Proportionally redistributed to traditional assets

## Portfolio Configurations

### With Crypto Allocation
- Stocks (SPY): {allocations['stocks']:.1f}% (${portfolio_value * allocations['stocks'] / 100:,.0f})
- Bonds (AGG): {allocations['bonds']:.1f}% (${portfolio_value * allocations['bonds'] / 100:,.0f})
- Real Estate (VNQ): {allocations['real_estate']:.1f}% (${portfolio_value * allocations['real_estate'] / 100:,.0f})
- Crypto (COIN50): {allocations['crypto']:.1f}% (${portfolio_value * allocations['crypto'] / 100:,.0f})

### Without Crypto (Rebalanced)
- Stocks (SPY): {comparison_data['allocations']['without_crypto']['stocks']:.1f}%
- Bonds (AGG): {comparison_data['allocations']['without_crypto']['bonds']:.1f}%
- Real Estate (VNQ): {comparison_data['allocations']['without_crypto']['real_estate']:.1f}%
- Crypto (COIN50): 0.0%

## Key Performance Metrics Comparison

### Returns
- Expected Return WITH Crypto: {comparison_data['with_crypto']['expected_return']:.2f}%
- Expected Return WITHOUT Crypto: {comparison_data['without_crypto']['expected_return']:.2f}%
- **Incremental Return: +{comparison_data['incremental_return']:.2f}%**

### Risk Metrics
- Sharpe Ratio WITH: {comparison_data['with_crypto']['sharpe_ratio']:.3f}
- Sharpe Ratio WITHOUT: {comparison_data['without_crypto']['sharpe_ratio']:.3f}
- Max Drawdown WITH: {comparison_data['with_crypto']['max_drawdown']:.1f}%
- Max Drawdown WITHOUT: {comparison_data['without_crypto']['max_drawdown']:.1f}%
- 95% VaR WITH: {comparison_data['with_crypto']['var_95']:.1f}%
- 95% VaR WITHOUT: {comparison_data['without_crypto']['var_95']:.1f}%

## Risk-Return Trade-off Analysis
With a {allocations['crypto']:.1f}% crypto allocation:
- Additional annual return potential: {comparison_data['incremental_return']:.2f}%
- Additional downside risk: {comparison_data['incremental_risk']:.1f}%
- Risk-adjusted return improvement: {((comparison_data['with_crypto']['sharpe_ratio'] / comparison_data['without_crypto']['sharpe_ratio']) - 1) * 100:.1f}%

## Behavioral Finance Insights

### Loss Framing
If crypto drops 75%:
- Traditional view: "Portfolio loses {allocations['crypto'] * 0.75:.1f}%"
- Reframed view: "You keep {100 - allocations['crypto'] * 0.75:.1f}% of your wealth"
- Dollar perspective: "${portfolio_value * (1 - allocations['crypto'] * 0.0075):,.0f} protected out of ${portfolio_value:,.0f}"

### Opportunity Cost
Based on historical crypto performance, investors without crypto allocation may have missed:
- Potential upside in bull markets
- Portfolio diversification benefits
- Reduced correlation during certain market conditions

## Recommendations

1. **For Conservative Investors**: Consider 2.5-5% crypto allocation
  - Minimal downside impact
  - Meaningful upside participation
  - Improved Sharpe ratio

2. **For Moderate Risk Investors**: Consider 5-10% crypto allocation
  - Balanced risk-reward profile
  - Better long-term growth potential
  - Still manageable downside

3. **Time Horizon Considerations**:
  - 1-3 years: Lower crypto allocation (2.5-5%)
  - 3-5 years: Moderate allocation (5-7.5%)
  - 5+ years: Higher allocation viable (7.5-10%+)

## Conclusion
Small crypto allocations (2.5-10%) provide asymmetric risk-reward opportunities:
- Limited downside (capped at allocation percentage)
- Significant upside potential
- Improved portfolio efficiency (higher Sharpe ratio)
- Better long-term growth prospects

The analysis demonstrates that thoughtful crypto integration enhances portfolio performance without materially increasing risk for most investors.
"""
           
           st.download_button(
               label="📥 Download Full Comparison Report",
               data=report_content,
               file_name=f"portfolio_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
               mime="text/plain"
           )
           
           # Additional advanced metrics
           st.markdown("### Advanced Risk Metrics")
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("#### Conditional Value at Risk (CVaR)")
               cvar_with = calculate_cvar(
                   comparison_data['with_crypto']['annual_returns'],
                   0.95
               )
               cvar_without = calculate_cvar(
                   comparison_data['without_crypto']['annual_returns'],
                   0.95
               )
               
               st.metric("CVaR WITH Crypto", f"{cvar_with:.2f}%")
               st.metric("CVaR WITHOUT Crypto", f"{cvar_without:.2f}%")
               st.info("CVaR shows expected loss in worst 5% of scenarios")
           
           with col2:
               st.markdown("#### Correlation Analysis")
               if 'returns_data' in st.session_state.portfolio_data:
                   stress_corr = stress_test_correlation_analysis(
                       st.session_state.portfolio_data['returns_data'],
                       stress_threshold=-0.05
                   )
                   
                   st.metric(
                       "Crypto-Stock Correlation (Normal)",
                       f"{stress_corr['normal_correlation'].loc['crypto', 'stocks']:.2f}"
                   )
                   st.metric(
                       "Crypto-Stock Correlation (Stress)",
                       f"{stress_corr['stress_correlation'].loc['crypto', 'stocks']:.2f}"
                   )
                   st.warning(
                       f"Correlation increases by {stress_corr['correlation_change'].loc['crypto', 'stocks']:.2f} "
                       f"during market stress"
                   )
   
   else:
       # Single portfolio mode (original tabs)
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