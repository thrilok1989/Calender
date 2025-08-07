import streamlit as st
import requests
import pandas as pd
import math
import numpy as np
from scipy.stats import norm
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Configure auto-refresh every 2 minutes
st_autorefresh(interval=2*60*1000, key="data_refresh")

# ===== Improved NSE Data Fetching =====
def fetch_nse_data(max_retries=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    try:
        # First get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        
        # Then fetch option chain
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Validate JSON response
        if not response.text.strip():
            raise ValueError("Empty response from NSE API")
            
        return response.json()
        
    except Exception as e:
        st.error(f"Failed to fetch NSE data after {max_retries} retries. Error: {str(e)}")
        st.warning("Please try again later or check your network connection.")
        return None

# ===== Greeks Calculation =====
def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta = (
        - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
        if option_type == 'CE'
        else
        - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    ) / 365
    rho = (
        K * T * math.exp(-r * T) * norm.cdf(d2)
        if option_type == 'CE'
        else -K * T * math.exp(-r * T) * norm.cdf(-d2)
    ) / 100
    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

# ===== Bias & Helper Functions =====
def delta_volume_bias(price_diff, volume_diff, chg_oi_diff):
    if price_diff > 0 and volume_diff > 0 and chg_oi_diff > 0:
        return "Bullish"
    elif price_diff < 0 and volume_diff > 0 and chg_oi_diff > 0:
        return "Bearish"
    elif price_diff > 0 and volume_diff > 0 and chg_oi_diff < 0:
        return "Bullish"
    elif price_diff < 0 and volume_diff > 0 and chg_oi_diff < 0:
        return "Bearish"
    else:
        return "Neutral"

def final_verdict(score):
    if score >= 4:
        return "Strong Bull"
    elif score >= 2:
        return "Bullish"
    elif score <= -4:
        return "Strong Bear"
    elif score <= -2:
        return "Bearish"
    else:
        return "Neutral"

# ===== Streamlit App =====
def main():
    st.set_page_config(layout="wide", page_title="Nifty Option Chain Analyzer")
    st.title("📊 Nifty Option Chain Bias Analyzer")
    
    with st.expander("ℹ️ About this App"):
        st.write("""
        This app analyzes NIFTY option chain data to detect bullish/bearish biases using:
        - Price action (LTP)
        - Open Interest changes
        - Volume analysis
        - Greeks (Delta, Gamma, Vega, Theta)
        - Implied Volatility
        """)
    
    # Fetch data with loading indicator
    with st.spinner("Fetching live NSE data..."):
        data = fetch_nse_data()
    
    if not data:
        st.stop()
    
    # Process data
    records = data["records"]["data"]
    expiry = data["records"]["expiryDates"][0]
    underlying = data["records"]["underlyingValue"]
    
    today = datetime.today()
    expiry_date = datetime.strptime(expiry, "%d-%b-%Y")
    T = max((expiry_date - today).days, 1) / 365
    r = 0.06  # Risk-free rate

    # Prepare CE and PE data
    calls, puts = [], []
    for item in records:
        if 'CE' in item and item['CE']['expiryDate'] == expiry:
            ce = item['CE']
            if ce['impliedVolatility'] > 0:
                ce.update(dict(zip(
                    ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                    calculate_greeks('CE', underlying, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                ))
            calls.append(ce)
        if 'PE' in item and item['PE']['expiryDate'] == expiry:
            pe = item['PE']
            if pe['impliedVolatility'] > 0:
                pe.update(dict(zip(
                    ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                    calculate_greeks('PE', underlying, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                ))
            puts.append(pe)

    # Create DataFrames and merge
    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE'))
    df = df.sort_values('strikePrice')

    # Filter ATM ±100 points
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df = df[df['strikePrice'].between(atm_strike - 100, atm_strike + 100)]
    df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')

    # Calculate biases
    results = []
    for _, row in df.iterrows():
        score = 0
        row_data = {
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
        }
        
        # Calculate all biases
        biases = {
            "LTP": "Bullish" if row['lastPrice_CE'] > row['lastPrice_PE'] else "Bearish",
            "OI": "Bearish" if row['openInterest_CE'] > row['openInterest_PE'] else "Bullish",
            "ChgOI": "Bearish" if row['changeinOpenInterest_CE'] > row['changeinOpenInterest_PE'] else "Bullish",
            "Volume": "Bullish" if row['totalTradedVolume_CE'] > row['totalTradedVolume_PE'] else "Bearish",
            "Delta": "Bullish" if row['Delta_CE'] > abs(row['Delta_PE']) else "Bearish",
            "Gamma": "Bullish" if row['Gamma_CE'] > row['Gamma_PE'] else "Bearish",
            "AskBid": "Bullish" if row['bidQty_CE'] > row['askQty_CE'] else "Bearish",
            "IV": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish"
        }
        
        # Add biases to row data
        for k, v in biases.items():
            row_data[f"{k}_Bias"] = v
            score += 1 if v == "Bullish" else -1

        # Calculate exposure metrics
        delta_exp_ce = row['Delta_CE'] * row['openInterest_CE']
        delta_exp_pe = row['Delta_PE'] * row['openInterest_PE']
        gamma_exp_ce = row['Gamma_CE'] * row['openInterest_CE']
        gamma_exp_pe = row['Gamma_PE'] * row['openInterest_PE']

        row_data.update({
            "DeltaExp": "Bullish" if delta_exp_ce > abs(delta_exp_pe) else "Bearish",
            "GammaExp": "Bullish" if gamma_exp_ce > gamma_exp_pe else "Bearish",
            "DVP_Bias": delta_volume_bias(
                row['lastPrice_CE'] - row['lastPrice_PE'],
                row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
            ),
            "Score": score,
            "Verdict": final_verdict(score),
            "Operator_Entry": "Entry Bull" if biases['OI'] == "Bullish" and biases['ChgOI'] == "Bullish" 
                            else "Entry Bear" if biases['OI'] == "Bearish" and biases['ChgOI'] == "Bearish" 
                            else "No Entry",
            "Action": "Scalp Bull" if score >= 4 else 
                     "Moment Bull" if score >= 2 else
                     "Scalp Bear" if score <= -4 else
                     "Moment Bear" if score <= -2 else "No Signal",
            "OI_Comparison": f"{round(row['openInterest_CE']/1e6, 2)}M vs {round(row['openInterest_PE']/1e6, 2)}M",
            "ChgOI_Comparison": f"{int(row['changeinOpenInterest_CE']/1000)}K vs {int(row['changeinOpenInterest_PE']/1000)}K"
        })

        results.append(row_data)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # ===== Display Results =====
    st.success(f"Data loaded successfully (Spot: {underlying:.2f}, Expiry: {expiry})")
    
    # Top Recommendation
    best = results_df.iloc[results_df['Score'].abs().idxmax()]
    reco_emoji = "🚀" if best['Score'] > 0 else "⚠️"
    st.subheader(f"{reco_emoji} Recommendation: {best['Action']} (Score: {best['Score']})")
    
    # Main Data Display
    cols_to_show = [
        "Strike", "Zone", "Verdict", "Score", "Action", 
        "OI_Comparison", "ChgOI_Comparison", "Operator_Entry"
    ]
    st.dataframe(
        results_df[cols_to_show].sort_values("Score", ascending=False),
        height=600,
        use_container_width=True
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Distribution")
        fig1 = go.Figure(data=[go.Histogram(x=results_df['Score'], nbinsx=20)])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Zone-wise Verdicts")
        verdict_counts = results_df.groupby(['Zone', 'Verdict']).size().unstack().fillna(0)
        st.bar_chart(verdict_counts)
    
    # Raw Data Explorer
    with st.expander("🔍 Raw Data Explorer"):
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()