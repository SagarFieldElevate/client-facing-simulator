"""
Configuration settings for the portfolio simulator
"""

# Pinecone settings
PINECONE_INDEX = "intelligence-main"
PINECONE_DIMENSION = 1536

# Asset mappings in Pinecone
ASSET_MAPPINGS = {
    'stocks': 'SPY Daily Close Price',
    'bonds': 'AGG Daily Close Price', 
    'real_estate': 'VNQ Daily Close Price',
    'crypto': 'COIN50 Perpetual Index (365 Days)'
}

# Default portfolio allocations
DEFAULT_ALLOCATIONS = {
    'stocks': 60.0,
    'bonds': 30.0,
    'real_estate': 5.0,
    'crypto': 5.0
}

# Simulation settings
DEFAULT_SIMULATIONS = 1000
MAX_SIMULATIONS = 10000
DEFAULT_TIME_HORIZON = 365
MAX_CRYPTO_ALLOCATION = 20.0

# Stress test scenarios
STRESS_SCENARIOS = {
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
    },
    "Dot-Com Bubble (2000)": {
        "stocks": -49,
        "bonds": 11,
        "real_estate": 15,
        "crypto": -80  # Hypothetical
    },
    "Black Monday (1987)": {
        "stocks": -22,
        "bonds": 3,
        "real_estate": -15,
        "crypto": -40  # Hypothetical
    }
}

# Chart colors
CHART_COLORS = {
    'primary': '#00d4ff',
    'secondary': '#0099ff',
    'success': '#00ff00',
    'warning': '#ffaa00',
    'danger': '#ff0000',
    'background': '#0a0a0a',
    'card_bg': 'rgba(255, 255, 255, 0.03)',
    'border': 'rgba(255, 255, 255, 0.1)'
}

# Risk levels
RISK_LEVELS = {
    'low': {'range': (0, 25), 'color': 'rgba(0, 255, 0, 0.3)'},
    'moderate': {'range': (25, 50), 'color': 'rgba(255, 255, 0, 0.3)'},
    'high': {'range': (50, 75), 'color': 'rgba(255, 165, 0, 0.3)'},
    'very_high': {'range': (75, 100), 'color': 'rgba(255, 0, 0, 0.3)'}
}