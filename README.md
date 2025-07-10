# Portfolio Risk Simulator

A client-facing portfolio allocation simulator that demonstrates how small crypto allocations (2.5-10%) impact traditional portfolio risk and return profiles. Built with Streamlit and real historical data from Pinecone.

## Features

- **Interactive Portfolio Builder**: Adjust allocations across stocks (SPY), bonds (AGG), real estate (VNQ), and crypto (COIN50)
- **Monte Carlo Simulations**: Run 100-10,000 simulations with configurable time horizons
- **Professional Visualizations**:
  - Portfolio growth fan charts
  - Return distribution histograms
  - Risk level gauges
  - Correlation heatmaps
- **Stress Test Scenarios**: See how your portfolio would perform in historical crises
- **Comprehensive Metrics**: Sharpe ratio, max drawdown, VaR, recovery time, and more
- **Dark Professional Theme**: Modern, mobile-responsive design

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd client-risk-simulator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.template .env
# Edit .env and add your Pinecone API key
```

## Configuration

### Pinecone Setup
The app requires access to Pinecone with the following data:
- Index: `intelligence-main`
- Vector dimensions: 1536

Required data in Pinecone:
- `SPY Daily Close Price` (S&P 500 ETF)
- `AGG Daily Close Price` (Bond ETF)
- `VNQ Daily Close Price` (Real Estate ETF)
- `COIN50 Perpetual Index (365 Days)` (Crypto index)

### Default Settings
Edit `config/settings.py` to modify:
- Default allocations (60% stocks, 30% bonds, 5% real estate, 5% crypto)
- Maximum crypto allocation (20%)
- Simulation parameters
- Stress test scenarios

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Use the sidebar to:
   - Set initial portfolio value
   - Adjust asset allocations (must sum to 100%)
   - Configure simulation parameters

4. Click "Run Portfolio Simulation" to generate results

5. Explore the tabs:
   - **Projections**: View portfolio growth scenarios
   - **Risk Analysis**: Examine risk metrics and distributions
   - **Stress Tests**: See performance in historical crises
   - **Report**: Download a comprehensive analysis

## Key Insights for Investors

The simulator demonstrates that:
- **Small crypto allocations (2.5-10%)** provide asymmetric upside potential
- **Limited downside risk**: If crypto drops 50%, a 5% allocation only impacts the portfolio by 2.5%
- **Improved Sharpe ratios** possible with proper diversification
- **Dollar impact visualization**: Shows actual dollar amounts at risk

## Project Structure

```
client-risk-simulator/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # API keys (create from .env.template)
├── .env.template              # Environment variable template
├── README.md                  # This file
├── utils/
│   ├── pinecone_client.py    # Pinecone data fetching
│   ├── portfolio_simulator.py # Monte Carlo simulation engine
│   ├── visualization.py      # Chart creation utilities
│   └── metrics.py            # Portfolio metrics calculations
├── config/
│   └── settings.py           # Application configuration
└── .streamlit/
    └── config.toml           # Streamlit theme configuration
```

## Technical Details

### Monte Carlo Simulation
- Uses GARCH(1,1) models for volatility modeling
- Implements correlated asset returns using Cholesky decomposition
- Generates realistic price paths based on historical patterns

### Data Processing
- Fetches real historical data from Pinecone
- Parses different data formats for each asset type
- Aligns data to common date ranges
- Calculates rolling correlations

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: 95% confidence interval for losses
- **Recovery Time**: Average days to recover from drawdowns

## Deployment

### Local Deployment
Follow the installation instructions above.

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables in Streamlit Cloud settings
4. Deploy

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Common Issues

1. **Pinecone connection error**:
   - Verify API key in `.env`
   - Check index name matches `intelligence-main`

2. **Missing data error**:
   - Ensure all required assets exist in Pinecone
   - Check data format matches expected patterns

3. **Simulation performance**:
   - Reduce number of simulations for faster results
   - Consider caching results for repeated analysis

## Future Enhancements

- [ ] Add more asset classes (commodities, international stocks)
- [ ] Implement rebalancing strategies
- [ ] Add tax impact analysis
- [ ] Include inflation adjustments
- [ ] Export results to PDF reports
- [ ] Add user authentication for saved portfolios

## New Features (Phase 1 & 2)

### Comparison Mode
Toggle between single portfolio analysis and side-by-side comparison of portfolios with and without crypto allocation.

### Protection vs Participation Analysis
Visualizes the trade-off between downside protection and upside participation for different crypto allocations, highlighting the "sweet spot" for optimal risk-adjusted returns.

### Time Horizon Analysis
Shows how optimal crypto allocation changes based on investment timeframe (1, 3, 5, and 10 years).

### Regret Minimization
Probability matrix showing the likelihood of regretting your allocation choice compared to alternatives.

### Break-Even Analysis
Calculates:
- How much crypto needs to gain to offset traditional losses
- How much crypto can drop before wiping out traditional gains

### Advanced Risk Metrics
- Conditional Value at Risk (CVaR) - Expected loss in worst scenarios
- Omega Ratio - Better metric for non-normal distributions
- Dynamic correlation analysis during market stress

### Behavioral Finance Features
- Loss framing alternatives (positive vs negative framing)
- Opportunity cost calculations
- Peer comparison insights

## Usage Tips

1. **Start with Comparison Mode OFF** to understand your current portfolio
2. **Enable Comparison Mode** to see the impact of adding/removing crypto
3. **Use the Protection vs Participation chart** to find your optimal allocation
4. **Check Time Horizon Analysis** to align allocation with your investment timeline
5. **Review Regret Analysis** to understand psychological aspects of your decision

## License

This project is proprietary software. All rights reserved.

## Support

For issues or questions, please contact the development team.