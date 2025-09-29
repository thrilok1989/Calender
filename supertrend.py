import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
import time

# Page config
st.set_page_config(page_title="Nifty Option Chain", layout="wide")

# Auto refresh every 1 minute
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

current_time = time.time()
if current_time - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = current_time
    st.rerun()

# Greeks Calculation
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

# Bias & Helper Functions
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

# Fetch NSE Option Chain
@st.cache_data(ttl=60)
def fetch_option_chain():
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    session.headers.update(headers)
    session.get("https://www.nseindia.com", timeout=5)
    
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    response = session.get(url, timeout=10)
    data = response.json()
    
    records = data["records"]["data"]
    expiry = data["records"]["expiryDates"][0]
    underlying = data["records"]["underlyingValue"]
    
    today = datetime.today()
    expiry_date = datetime.strptime(expiry, "%d-%b-%Y")
    T = max((expiry_date - today).days, 1) / 365
    r = 0.06
    
    calls, puts = [], []
    for item in records:
        if 'CE' in item and item['CE']['expiryDate'] == expiry:
            ce = item['CE']
            if ce['impliedVolatility'] > 0:
                ce.update(dict(zip(
                    ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                    calculate_greeks('CE', underlying, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                )))
            calls.append(ce)
        if 'PE' in item and item['PE']['expiryDate'] == expiry:
            pe = item['PE']
            if pe['impliedVolatility'] > 0:
                pe.update(dict(zip(
                    ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
                    calculate_greeks('PE', underlying, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                )))
            puts.append(pe)
    
    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE'))
    df = df.sort_values('strikePrice')
    
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df = df[df['strikePrice'].between(atm_strike - 100, atm_strike + 100)]
    df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
    
    return df, underlying, atm_strike

# Main App
st.title("🔥 Nifty Option Chain Bias Summary")

# Display last update time
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: Every 60 seconds")

try:
    df, underlying, atm_strike = fetch_option_chain()
    
    results = []
    for _, row in df.iterrows():
        score = 0
        row_data = {
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
        }
        
        row_data["LTP_Bias"] = "Bullish" if row['lastPrice_CE'] > row['lastPrice_PE'] else "Bearish"
        row_data["OI_Bias"] = "Bearish" if row['openInterest_CE'] > row['openInterest_PE'] else "Bullish"
        row_data["ChgOI_Bias"] = "Bearish" if row['changeinOpenInterest_CE'] > row['changeinOpenInterest_PE'] else "Bullish"
        row_data["Volume_Bias"] = "Bullish" if row['totalTradedVolume_CE'] > row['totalTradedVolume_PE'] else "Bearish"
        row_data["Delta_Bias"] = "Bullish" if row['Delta_CE'] > abs(row['Delta_PE']) else "Bearish"
        row_data["Gamma_Bias"] = "Bullish" if row['Gamma_CE'] > row['Gamma_PE'] else "Bearish"
        row_data["AskBid_Bias"] = "Bullish" if row['bidQty_CE'] > row['askQty_CE'] else "Bearish"
        row_data["IV_Bias"] = "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish"
        
        delta_exp_ce = row['Delta_CE'] * row['openInterest_CE']
        delta_exp_pe = row['Delta_PE'] * row['openInterest_PE']
        gamma_exp_ce = row['Gamma_CE'] * row['openInterest_CE']
        gamma_exp_pe = row['Gamma_PE'] * row['openInterest_PE']
        
        row_data["DeltaExp_Bias"] = "Bullish" if delta_exp_ce > abs(delta_exp_pe) else "Bearish"
        row_data["GammaExp_Bias"] = "Bullish" if gamma_exp_ce > gamma_exp_pe else "Bearish"
        row_data["DVP_Bias"] = delta_volume_bias(
            row['lastPrice_CE'] - row['lastPrice_PE'],
            row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
            row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
        )
        
        for k in row_data:
            if "_Bias" in k:
                score += 1 if row_data[k] == "Bullish" else -1
        
        row_data["Score"] = score
        row_data["Verdict"] = final_verdict(score)
        row_data["Operator Entry"] = "Entry Bull" if row_data['OI_Bias'] == "Bullish" and row_data['ChgOI_Bias'] == "Bullish" else ("Entry Bear" if row_data['OI_Bias'] == "Bearish" and row_data['ChgOI_Bias'] == "Bearish" else "No Entry")
        row_data["Scalp/Moment"] = "Scalp Bull" if score >= 4 else ("Moment Bull" if score >= 2 else ("Scalp Bear" if score <= -4 else ("Moment Bear" if score <= -2 else "No Signal")))
        row_data["FakeReal"] = "Real Up" if score >= 4 else ("Fake Up" if 1 <= score < 4 else ("Real Down" if score <= -4 else ("Fake Down" if -4 < score <= -1 else "No Move")))
        
        row_data["ChgOI (C vs P)"] = f"{int(row['changeinOpenInterest_CE']/1000)}K {'>' if row['changeinOpenInterest_CE'] > row['changeinOpenInterest_PE'] else '<' if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else '≈'} {int(row['changeinOpenInterest_PE']/1000)}K"
        row_data["OI (C vs P)"] = f"{round(row['openInterest_CE']/1e6, 2)}M {'>' if row['openInterest_CE'] > row['openInterest_PE'] else '<' if row['openInterest_CE'] < row['openInterest_PE'] else '≈'} {round(row['openInterest_PE']/1e6, 2)}M"
        
        results.append(row_data)
    
    best = max(results, key=lambda x: abs(x['Score']))
    st.success(f"📢 TRADE {'CALL' if best['Score'] > 0 else 'PUT'} | Momentum: {'STRONG' if abs(best['Score']) >= 4 else 'MODERATE'} | Move: {best['FakeReal'].upper()} | Suggested: {best['Scalp/Moment'].upper()}")
    
    st.subheader("📊 Complete Bias Analysis (ATM ±2 Strikes)")
    
    cols = ["Strike", "Zone", "LTP_Bias", "OI_Bias", "ChgOI_Bias", "Volume_Bias", "Delta_Bias", 
            "Gamma_Bias", "AskBid_Bias", "IV_Bias", "DeltaExp_Bias", "GammaExp_Bias", "DVP_Bias",
            "Score", "Verdict", "Operator Entry", "Scalp/Moment", "FakeReal", "ChgOI (C vs P)", "OI (C vs P)"]
    
    display_df = pd.DataFrame(results)[cols]
    
    def color_bias(val):
        if val == "Bullish":
            return 'background-color: #90EE90'
        elif val == "Bearish":
            return 'background-color: #FFB6C1'
        elif val == "Neutral":
            return 'background-color: #FFFFE0'
        return ''
    
    styled_df = display_df.style.applymap(color_bias, subset=[col for col in cols if '_Bias' in col or col == 'DVP_Bias'])
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
except Exception as e:
    st.error(f"Error: {str(e)}")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
