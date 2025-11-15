import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis
import warnings

# Ignore minor warnings during execution
warnings.filterwarnings('ignore')

# --- Configuration & Styling ---
DATA_FILE = 'microsoft_stock.csv'
st.set_page_config(
    page_title="Microsoft Probability & Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Data Loading and Preprocessing (Cached for Performance) ---
@st.cache_data
def load_data(file_path):
    """Loads and preprocesses stock data."""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.dropna(subset=['Adj Close'])
        
        # Calculate daily returns
        df['Returns'] = df['Adj Close'].pct_change().dropna()
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please place it in the same directory.")
        return None

# --- 2. Core Probability and Statistics Functions (Cached) ---
@st.cache_data
def calculate_metrics(returns):
    """Calculates key statistical and probability metrics."""
    mu = returns.mean()
    sigma = returns.std()
    
    # Calculate key probabilities based on historical data
    prob_positive = (returns > 0).mean()
    prob_negative = (returns < 0).mean()
    
    # Check for tail risk (e.g., > 1% change)
    prob_large_positive = (returns > 0.01).mean()
    prob_large_negative = (returns < -0.01).mean()
    
    return {
        'mu': mu, 
        'sigma': sigma,
        'skewness': skew(returns),
        'kurtosis': kurtosis(returns),
        'prob_positive': prob_positive,
        'prob_negative': prob_negative,
        'prob_large_positive': prob_large_positive,
        'prob_large_negative': prob_large_negative,
    }

# --- 3. Risk Metrics (VaR/CVaR) Functions (Cached) ---
@st.cache_data
def calculate_risk_metrics(returns, confidence_level=0.95):
    """Calculates Value-at-Risk (VaR) and Conditional VaR (CVaR)."""
    
    # Historical VaR: The quantile of the historical returns distribution
    var_historical = returns.quantile(1 - confidence_level)
    
    # Parametric VaR (assuming Normal Distribution)
    z_score = norm.ppf(1 - confidence_level) 
    var_parametric = returns.mean() + returns.std() * z_score
    
    # Historical CVaR (Expected Shortfall): Mean of returns below the VaR threshold
    cvar_historical = returns[returns < var_historical].mean()
    
    return {
        'var_hist': var_historical,
        'var_param': var_parametric,
        'cvar_hist': cvar_historical
    }

# --- 4. Conditional Probability / Momentum Functions (Cached) ---
@st.cache_data
def calculate_conditional_prob(returns):
    """Calculates momentum and mean reversion probabilities."""
    
    positive_returns = returns > 0
    negative_returns = returns < 0
    
    prev_positive = positive_returns.shift(1).dropna()
    prev_negative = negative_returns.shift(1).dropna()
    
    current_returns = returns.iloc[1:]

    # Momentum: P(Gain | Previous Gain)
    prob_pos_given_pos = (current_returns[prev_positive] > 0).mean()
    
    # Momentum: P(Loss | Previous Loss)
    prob_neg_given_neg = (current_returns[prev_negative] < 0).mean()

    # Mean Reversion: P(Gain | Previous Loss)
    prob_pos_given_neg = (current_returns[prev_negative] > 0).mean()
    
    return {
        'P(G|G)': prob_pos_given_pos,
        'P(L|L)': prob_neg_given_neg,
        'P(G|L)': prob_pos_given_neg
    }

# --- 5. Monte Carlo Simulation Function (Cached) ---
@st.cache_data(show_spinner="Running Monte Carlo Simulation...")
def run_monte_carlo(current_price, mu_log, sigma_log, forecast_days, num_paths):
    """Performs the Geometric Brownian Motion Monte Carlo simulation."""
    dt = 1/252 # Daily step based on ~252 trading days per year
    
    sim_paths = np.zeros((forecast_days, num_paths))
    sim_paths[0] = current_price
    
    for t in range(1, forecast_days):
        # Calculate the next price step using Geometric Brownian Motion formula
        Z = np.random.standard_normal(num_paths)
        sim_paths[t] = sim_paths[t-1] * np.exp((mu_log - 0.5 * sigma_log**2) * dt + sigma_log * np.sqrt(dt) * Z)
        
    return sim_paths

# --- Main Application Body ---

def main():
    st.title("Microsoft Stock Probability and Forecasting Dashboard")
    
    data = load_data(DATA_FILE)
    if data is None:
        return

    returns = data['Returns']
    metrics = calculate_metrics(returns)
    current_price = data['Adj Close'].iloc[-1]
    
    # Calculate new metrics
    risk_metrics = calculate_risk_metrics(returns, confidence_level=0.95)
    conditional_prob = calculate_conditional_prob(returns)
    
    # --- Sidebar for Inputs ---
    st.sidebar.title("Configuration")
    st.sidebar.subheader("Monte Carlo Inputs")
    forecast_days = st.sidebar.slider("Forecast Period (Trading Days)", 
                                       min_value=30, max_value=500, value=252, step=10)
    num_paths = st.sidebar.slider("Number of Paths (Simulation Runs)", 
                                   min_value=10, max_value=500, value=100, step=10)
    
    # --- Row 1: Core Metrics ---
    st.header("1. Core Probability Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Current Price", f"${current_price:,.2f}")
    col2.metric("Mean Daily Return ($\mu$)", f"{metrics['mu'] * 100:.4f}%")
    col3.metric("Std. Dev. ($\sigma$)", f"{metrics['sigma'] * 100:.4f}%")
    col4.metric("Prob. of Gain", f"{metrics['prob_positive'] * 100:.2f}%")
    col5.metric("Prob. of Loss", f"{metrics['prob_negative'] * 100:.2f}%")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Skewness", f"{metrics['skewness']:.4f}", 
                  help="Positive indicates more small gains/fewer large losses; Negative is the opposite.")
    col_s2.metric("Kurtosis", f"{metrics['kurtosis']:.4f}", 
                  help="High kurtosis (above 0) indicates 'fat tails', meaning extreme returns are more likely.")
    col_s3.metric("Prob. > 1% Gain", f"{metrics['prob_large_positive'] * 100:.2f}%")
    col_s4.metric("Prob. < 1% Loss", f"{metrics['prob_large_negative'] * 100:.2f}%")

    st.markdown("---")

    # --- New Row: Risk and Momentum ---
    st.header("2. Risk Metrics & Market Behavior")
    
    # VaR/CVaR Sub-section
    st.subheader("Downside Risk (95% Confidence)")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    
    risk_col1.metric("Historical VaR", f"{risk_metrics['var_hist'] * 100:.2f}%",
                     help="Maximum expected loss at the 95% confidence level based on historical data.")
    risk_col2.metric("Parametric VaR", f"{risk_metrics['var_param'] * 100:.2f}%",
                     help="Maximum expected loss assuming a Normal distribution.")
    risk_col3.metric("Historical CVaR", f"{risk_metrics['cvar_hist'] * 100:.2f}%",
                     help="Expected loss given that VaR is exceeded (the mean of the worst 5% of returns).")
    
    # Momentum Sub-section
    st.subheader("Market Momentum (Conditional Probability)")
    momentum_col1, momentum_col2, momentum_col3, momentum_col4 = st.columns(4)
    
    momentum_col1.metric("P(Gain | Prev. Gain)", f"{conditional_prob['P(G|G)'] * 100:.2f}%",
                         help="Probability of a gain today, given the stock gained yesterday (Positive Momentum).")
    momentum_col2.metric("P(Loss | Prev. Loss)", f"{conditional_prob['P(L|L)'] * 100:.2f}%",
                         help="Probability of a loss today, given the stock lost yesterday (Negative Momentum).")
    momentum_col3.metric("P(Gain | Prev. Loss)", f"{conditional_prob['P(G|L)'] * 100:.2f}%",
                         help="Probability of a gain today, given the stock lost yesterday (Mean Reversion tendency).")
    
    st.markdown("---")
    
    # --- Row 3: Visualizations ---
    st.header("3. Historical Analysis & Distribution")
    col_chart1, col_chart2 = st.columns(2)

    # Plot 1: Historical Price
    with col_chart1:
        st.subheader("Historical Adjusted Close Price")
        fig_price = px.line(data.reset_index(), 
                            x='Date', 
                            y='Adj Close', 
                            template='plotly_white')
        st.plotly_chart(fig_price, use_container_width=True)

    # Plot 2: Returns Histogram
    with col_chart2:
        st.subheader("Daily Returns Distribution vs. Normal PDF")
        
        fig_hist = go.Figure(data=[go.Histogram(
            x=returns,
            name='Daily Returns',
            histnorm='probability density',
            marker_color='#007ACC',
            nbinsx=100
        )])
        
        # Overlay Normal Distribution (PDF)
        x_range = np.linspace(returns.min(), returns.max(), 200)
        pdf = norm.pdf(x_range, metrics['mu'], metrics['sigma'])
        
        fig_hist.add_trace(go.Scatter(
            x=x_range,
            y=pdf,
            mode='lines',
            name='Normal PDF',
            line={'color': 'red', 'width': 2}
        ))
        
        fig_hist.update_layout(
            xaxis_title='Daily Return',
            yaxis_title='Probability Density',
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")

    # --- Row 4: Monte Carlo Simulation (Forecasting) ---
    st.header("4. Interactive Monte Carlo Price Forecast")
    
    # Calculate log returns for GBM
    log_returns = np.log(1 + returns).dropna()
    mu_log = log_returns.mean()
    sigma_log = log_returns.std()
    
    sim_paths = run_monte_carlo(current_price, mu_log, sigma_log, forecast_days, num_paths)
    
    # Plotting Monte Carlo
    fig_mc = go.Figure()
    
    for i in range(num_paths):
        fig_mc.add_trace(go.Scatter(
            y=sim_paths[:, i], 
            mode='lines', 
            opacity=0.2,
            line={'width': 1},
            name=f'Path {i+1}'
        ))
    
    # Calculate and plot the mean path and confidence interval
    final_prices = pd.Series(sim_paths[-1, :])
    mean_path = sim_paths.mean(axis=1)
    upper_bound = np.percentile(sim_paths, 97.5, axis=1) # 97.5 percentile
    lower_bound = np.percentile(sim_paths, 2.5, axis=1)  # 2.5 percentile
    
    fig_mc.add_trace(go.Scatter(
        y=mean_path, 
        mode='lines', 
        name='Expected Price (Mean)', 
        line={'color': 'black', 'width': 3}
    ))
    
    # Confidence bounds (shaded area)
    fig_mc.add_trace(go.Scatter(
        y=upper_bound, 
        mode='lines', 
        line=dict(width=0), 
        showlegend=False
    ))
    fig_mc.add_trace(go.Scatter(
        y=lower_bound, 
        fill='tonexty', 
        fillcolor='rgba(0, 122, 204, 0.1)', 
        mode='lines', 
        line=dict(width=0), 
        name='95% Confidence Interval'
    ))

    fig_mc.update_layout(
        xaxis_title='Trading Days from Today',
        yaxis_title='Price (USD)',
        template='plotly_white',
        showlegend=False,
        title=f'Monte Carlo Simulation: {num_paths} Paths over {forecast_days} Days'
    )
    st.plotly_chart(fig_mc, use_container_width=True)
    
    # Monte Carlo Summary Metrics
    prob_above_current = (final_prices > current_price).mean()
    
    st.subheader("Monte Carlo Forecast Summary")
    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
    mc_col1.metric("Expected Final Price", f"${final_prices.mean():,.2f}")
    mc_col2.metric("Prob. Price Rises", f"{prob_above_current * 100:.2f}%")
    mc_col3.metric("95% Low Price (VaR)", f"${final_prices.quantile(0.025):,.2f}", 
                   help="The 2.5th percentile of all simulated final prices.")
    mc_col4.metric("95% High Price", f"${final_prices.quantile(0.975):,.2f}", 
                   help="The 97.5th percentile of all simulated final prices.")


if __name__ == '__main__':
    main()