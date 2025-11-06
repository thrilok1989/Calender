import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
import time
import pytz

# Page config
st.set_page_config(page_title="Nifty Option Chain", layout="wide")

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    return datetime.now(IST)

def is_trading_hours():
    now = get_ist_time()
    # Monday = 0, Friday = 4
    if now.weekday() > 4:  # Saturday(5) or Sunday(6)
        return False
    
    current_time = now.time()
    market_open = datetime.strptime("09:00", "%H:%M").time()
    market_close = datetime.strptime("15:45", "%H:%M").time()
    
    return market_open <= current_time <= market_close

# Telegram Configuration
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except:
            pass

# Auto refresh every 1 minute during trading hours
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
    st.session_state.last_alert = {}

current_time = time.time()
if current_time - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = current_time
    if is_trading_hours():
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

def calculate_pcr_safe(put_value, call_value):
    """
    Safely calculate PCR handling negative values
    PCR = PUT / CALL (using absolute values for negative change in OI)
    """
    if call_value == 0:
        return 0.0
    
    # For Change in OI which can be negative, use absolute values
    if put_value < 0 or call_value < 0:
        return abs(put_value) / abs(call_value)
    
    return put_value / call_value

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
    df_all = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE'))
    df_all = df_all.sort_values('strikePrice')
    
    # Calculate ATM as nearest strike to spot price
    atm_strike = round(underlying / 50) * 50
    
    # Filter ATM ±2 strikes (±100 points for Nifty which has 50 point intervals)
    df = df_all[df_all['strikePrice'].between(atm_strike - 100, atm_strike + 100)].copy()
    df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < atm_strike else 'OTM')
    
    # Filter ATM ±10 strikes (±500 points)
    df_wide = df_all[df_all['strikePrice'].between(atm_strike - 500, atm_strike + 500)].copy()
    df_wide['Zone'] = df_wide['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < atm_strike else 'OTM')
    
    return df, df_wide, underlying, atm_strike

# Main App
st.title("🔥 Nifty Option Chain Bias Summary")

# Display last update time
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: Every 60 seconds")

# Telegram status
with st.sidebar:
    st.header("⚙️ Telegram Alerts")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.success("✅ Telegram Connected")
        st.info("Alerts trigger when:\n- Bull: Score > 10\n- Bear: Score < -10\n- All ATM biases align")
    else:
        st.warning("⚠️ Telegram Not Configured")
        st.code("Add to .streamlit/secrets.toml:\nTELEGRAM_BOT_TOKEN = 'your_token'\nTELEGRAM_CHAT_ID = 'your_chat_id'")

try:
    df, df_wide, underlying, atm_strike = fetch_option_chain()
    
    # Calculate total Change in OI for ATM ±2 strikes
    total_chg_oi_ce = df['changeinOpenInterest_CE'].sum()
    total_chg_oi_pe = df['changeinOpenInterest_PE'].sum()
    chg_oi_diff = total_chg_oi_ce - total_chg_oi_pe
    chg_oi_diff_pct = (chg_oi_diff / abs(total_chg_oi_pe) * 100) if total_chg_oi_pe != 0 else 0
    
    # Calculate total OI for Call vs Put (ATM ±2)
    total_oi_ce = df['openInterest_CE'].sum()
    total_oi_pe = df['openInterest_PE'].sum()
    oi_diff = total_oi_ce - total_oi_pe
    oi_diff_pct = (oi_diff / total_oi_pe * 100) if total_oi_pe != 0 else 0
    
    # Calculate total Volume for Call vs Put (ATM ±2)
    total_vol_ce = df['totalTradedVolume_CE'].sum()
    total_vol_pe = df['totalTradedVolume_PE'].sum()
    vol_diff = total_vol_ce - total_vol_pe
    vol_diff_pct = (vol_diff / total_vol_pe * 100) if total_vol_pe != 0 else 0
    
    # Calculate total Change in OI for ATM ±10 strikes (WIDER RANGE)
    total_chg_oi_ce_wide = df_wide['changeinOpenInterest_CE'].sum()
    total_chg_oi_pe_wide = df_wide['changeinOpenInterest_PE'].sum()
    chg_oi_diff_wide = total_chg_oi_ce_wide - total_chg_oi_pe_wide
    chg_oi_diff_pct_wide = (chg_oi_diff_wide / abs(total_chg_oi_pe_wide) * 100) if total_chg_oi_pe_wide != 0 else 0
    
    # Calculate total OI for Call vs Put (ATM ±10)
    total_oi_ce_wide = df_wide['openInterest_CE'].sum()
    total_oi_pe_wide = df_wide['openInterest_PE'].sum()
    oi_diff_wide = total_oi_ce_wide - total_oi_pe_wide
    oi_diff_pct_wide = (oi_diff_wide / total_oi_pe_wide * 100) if total_oi_pe_wide != 0 else 0
    
    # Calculate total Volume for Call vs Put (ATM ±10)
    total_vol_ce_wide = df_wide['totalTradedVolume_CE'].sum()
    total_vol_pe_wide = df_wide['totalTradedVolume_PE'].sum()
    vol_diff_wide = total_vol_ce_wide - total_vol_pe_wide
    vol_diff_pct_wide = (vol_diff_wide / total_vol_pe_wide * 100) if total_vol_pe_wide != 0 else 0
    
    # Determine overall bias from Change in OI (ATM ±2)
    chg_oi_bias = "BEARISH 🔴" if chg_oi_diff > 0 else "BULLISH 🟢" if chg_oi_diff < 0 else "NEUTRAL ⚪"
    oi_bias = "BEARISH 🔴" if oi_diff > 0 else "BULLISH 🟢" if oi_diff < 0 else "NEUTRAL ⚪"
    vol_bias = "BEARISH 🔴" if vol_diff > 0 else "BULLISH 🟢" if vol_diff < 0 else "NEUTRAL ⚪"
    
    # Determine overall bias from Change in OI (ATM ±10)
    chg_oi_bias_wide = "BEARISH 🔴" if chg_oi_diff_wide > 0 else "BULLISH 🟢" if chg_oi_diff_wide < 0 else "NEUTRAL ⚪"
    oi_bias_wide = "BEARISH 🔴" if oi_diff_wide > 0 else "BULLISH 🟢" if oi_diff_wide < 0 else "NEUTRAL ⚪"
    vol_bias_wide = "BEARISH 🔴" if vol_diff_wide > 0 else "BULLISH 🟢" if vol_diff_wide < 0 else "NEUTRAL ⚪"
    
    # Display key metrics at top with enhanced OI analysis
    st.subheader("📊 Market Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Spot Price", f"₹{underlying:,.2f}")
    with col2:
        st.metric("ATM Strike", f"{atm_strike}")
    with col3:
        st.metric("Analysis Range", f"±2 & ±10 Strikes")
    
    # Create tabs for different strike ranges
    tab1, tab2 = st.tabs(["📍 ATM ±2 Strikes (Focused)", "📊 ATM ±10 Strikes (Broader)"])
    
    with tab1:
        st.info(f"**Strike Range:** {atm_strike-100} to {atm_strike+100} (5 strikes)")
        
        # Total Change in OI Analysis (ATM ±2)
        st.subheader("🔥 Total Change in OI Analysis (ATM ±2 Strikes)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Chg OI (CALL)", 
                f"{int(total_chg_oi_ce/1000):,}K",
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Chg OI (PUT)", 
                f"{int(total_chg_oi_pe/1000):,}K",
                delta=None
            )
        
        with col3:
            st.metric(
                "Difference", 
                f"{int(chg_oi_diff/1000):,}K",
                delta=f"{chg_oi_diff_pct:+.2f}%"
            )
        
        with col4:
            st.metric(
                "Change OI Bias", 
                chg_oi_bias.split()[0],
                delta=None
            )
        
        # Additional OI and Volume metrics (ATM ±2)
        st.subheader("📈 Total OI & Volume Comparison (ATM ±2)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Total Open Interest**")
            st.metric("CALL OI", f"{round(total_oi_ce/1e6, 2)}M")
            st.metric("PUT OI", f"{round(total_oi_pe/1e6, 2)}M")
            st.metric("OI Bias", oi_bias.split()[0], delta=f"{oi_diff_pct:+.2f}%")
        
        with col2:
            st.markdown("**Total Volume**")
            st.metric("CALL Volume", f"{int(total_vol_ce/1000):,}K")
            st.metric("PUT Volume", f"{int(total_vol_pe/1000):,}K")
            st.metric("Volume Bias", vol_bias.split()[0], delta=f"{vol_diff_pct:+.2f}%")
        
        with col3:
            st.markdown("**PCR Ratios**")
            pcr_oi = calculate_pcr_safe(total_oi_pe, total_oi_ce)
            pcr_volume = calculate_pcr_safe(total_vol_pe, total_vol_ce)
            pcr_chg_oi = calculate_pcr_safe(total_chg_oi_pe, total_chg_oi_ce)
            
            # Color code PCR values
            pcr_oi_color = "🟢" if pcr_oi > 1.5 else "🔴" if pcr_oi < 0.7 else "⚪"
            pcr_vol_color = "🟢" if pcr_volume > 1.5 else "🔴" if pcr_volume < 0.7 else "⚪"
            pcr_chg_color = "🟢" if pcr_chg_oi > 1.5 else "🔴" if pcr_chg_oi < 0.7 else "⚪"
            
            st.metric("PCR (OI)", f"{pcr_oi:.3f} {pcr_oi_color}")
            st.metric("PCR (Volume)", f"{pcr_volume:.3f} {pcr_vol_color}")
            st.metric("PCR (Chg OI)", f"{pcr_chg_oi:.3f} {pcr_chg_color}")
            
            # Add note about negative change in OI
            if total_chg_oi_ce < 0 or total_chg_oi_pe < 0:
                st.caption("⚠️ Negative Chg OI detected - using absolute values for PCR")
    
    with tab2:
        st.info(f"**Strike Range:** {atm_strike-500} to {atm_strike+500} (21 strikes)")
        
        # Total Change in OI Analysis (ATM ±10)
        st.subheader("🔥 Total Change in OI Analysis (ATM ±10 Strikes)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Chg OI (CALL)", 
                f"{int(total_chg_oi_ce_wide/1000):,}K",
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Chg OI (PUT)", 
                f"{int(total_chg_oi_pe_wide/1000):,}K",
                delta=None
            )
        
        with col3:
            st.metric(
                "Difference", 
                f"{int(chg_oi_diff_wide/1000):,}K",
                delta=f"{chg_oi_diff_pct_wide:+.2f}%"
            )
        
        with col4:
            st.metric(
                "Change OI Bias", 
                chg_oi_bias_wide.split()[0],
                delta=None
            )
        
        # Additional OI and Volume metrics (ATM ±10)
        st.subheader("📈 Total OI & Volume Comparison (ATM ±10)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Total Open Interest**")
            st.metric("CALL OI", f"{round(total_oi_ce_wide/1e6, 2)}M")
            st.metric("PUT OI", f"{round(total_oi_pe_wide/1e6, 2)}M")
            st.metric("OI Bias", oi_bias_wide.split()[0], delta=f"{oi_diff_pct_wide:+.2f}%")
        
        with col2:
            st.markdown("**Total Volume**")
            st.metric("CALL Volume", f"{int(total_vol_ce_wide/1000):,}K")
            st.metric("PUT Volume", f"{int(total_vol_pe_wide/1000):,}K")
            st.metric("Volume Bias", vol_bias_wide.split()[0], delta=f"{vol_diff_pct_wide:+.2f}%")
        
        with col3:
            st.markdown("**PCR Ratios**")
            pcr_oi_wide = calculate_pcr_safe(total_oi_pe_wide, total_oi_ce_wide)
            pcr_volume_wide = calculate_pcr_safe(total_vol_pe_wide, total_vol_ce_wide)
            pcr_chg_oi_wide = calculate_pcr_safe(total_chg_oi_pe_wide, total_chg_oi_ce_wide)
            
            # Color code PCR values
            pcr_oi_wide_color = "🟢" if pcr_oi_wide > 1.5 else "🔴" if pcr_oi_wide < 0.7 else "⚪"
            pcr_vol_wide_color = "🟢" if pcr_volume_wide > 1.5 else "🔴" if pcr_volume_wide < 0.7 else "⚪"
            pcr_chg_wide_color = "🟢" if pcr_chg_oi_wide > 1.5 else "🔴" if pcr_chg_oi_wide < 0.7 else "⚪"
            
            st.metric("PCR (OI)", f"{pcr_oi_wide:.3f} {pcr_oi_wide_color}")
            st.metric("PCR (Volume)", f"{pcr_volume_wide:.3f} {pcr_vol_wide_color}")
            st.metric("PCR (Chg OI)", f"{pcr_chg_oi_wide:.3f} {pcr_chg_wide_color}")
            
            # Add note about negative change in OI
            if total_chg_oi_ce_wide < 0 or total_chg_oi_pe_wide < 0:
                st.caption("⚠️ Negative Chg OI detected - using absolute values for PCR")
    
    # Interpretation guide
    with st.expander("📖 How to Interpret Change in OI Analysis"):
        st.markdown("""
        **Change in OI (Open Interest) Interpretation:**
        
        - **CALL Chg OI > PUT Chg OI** = BEARISH 🔴
          - More Call writing (resistance) or Call buying (if prices rising)
          - Generally indicates bearish sentiment or strong resistance
        
        - **PUT Chg OI > CALL Chg OI** = BULLISH 🟢
          - More Put writing (support) or Put buying (if prices falling)
          - Generally indicates bullish sentiment or strong support
        
        **Understanding Negative Change in OI:**
        - Negative Change in OI means positions are being closed/unwound
        - When calculating PCR for Change in OI, we use absolute values if either is negative
        - This ensures PCR remains meaningful (always positive)
        
        **PCR (Put-Call Ratio) Guide:**
        - **PCR > 1.5** 🟢 = Bullish (More Put activity - potential support)
        - **PCR 0.7-1.5** ⚪ = Neutral (Balanced activity)
        - **PCR < 0.7** 🔴 = Bearish (More Call activity - potential resistance)
        
        **ATM ±2 vs ATM ±10:**
        - **ATM ±2 (5 strikes)**: More focused, immediate support/resistance levels
        - **ATM ±10 (21 strikes)**: Broader market sentiment, overall directional bias
        
        **Best Strategy:**
        - Use ATM ±2 for precise entry/exit points
        - Use ATM ±10 for overall market direction confirmation
        - Look for alignment between both ranges for stronger signals
        """)
    
    # Comparison between ranges
    st.subheader("🔄 Range Comparison (ATM ±2 vs ATM ±10)")
    
    comparison_data = {
        "Metric": ["Change OI Bias", "OI Bias", "Volume Bias", "PCR (OI)", "PCR (Volume)", "PCR (Chg OI)"],
        "ATM ±2 Strikes": [
            chg_oi_bias.split()[0],
            oi_bias.split()[0],
            vol_bias.split()[0],
            f"{pcr_oi:.3f}",
            f"{pcr_volume:.3f}",
            f"{pcr_chg_oi:.3f}"
        ],
        "ATM ±10 Strikes": [
            chg_oi_bias_wide.split()[0],
            oi_bias_wide.split()[0],
            vol_bias_wide.split()[0],
            f"{pcr_oi_wide:.3f}",
            f"{pcr_volume_wide:.3f}",
            f"{pcr_chg_oi_wide:.3f}"
        ],
        "Alignment": [
            "✅" if chg_oi_bias == chg_oi_bias_wide else "❌",
            "✅" if oi_bias == oi_bias_wide else "❌",
            "✅" if vol_bias == vol_bias_wide else "❌",
            "✅" if abs(pcr_oi - pcr_oi_wide) < 0.2 else "❌",
            "✅" if abs(pcr_volume - pcr_volume_wide) < 0.2 else "❌",
            "✅" if abs(pcr_chg_oi - pcr_chg_oi_wide) < 0.2 else "❌"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    results = []
    for _, row in df.iterrows():
        score = 0
        row_data = {
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
        }
        
        row_data["OI_Bias"] = "Bearish" if row['openInterest_CE'] > row['openInterest_PE'] else "Bullish"
        row_data["ChgOI_Bias"] = "Bearish" if row['changeinOpenInterest_CE'] > row['changeinOpenInterest_PE'] else "Bullish"
        row_data["Volume_Bias"] = "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish"
        row_data["Delta_Bias"] = "Bullish" if abs(row.get('Delta_CE', 0)) > abs(row.get('Delta_PE', 0)) else "Bearish"
        row_data["Gamma_Bias"] = "Bullish" if row.get('Gamma_CE', 0) > row.get('Gamma_PE', 0) else "Bearish"
        row_data["AskBid_Bias"] = "Bullish" if row.get('bidQty_CE', 0) > row.get('askQty_CE', 1) else "Bearish"
        row_data["IV_Bias"] = "Bullish" if row.get('impliedVolatility_CE', 0) < row.get('impliedVolatility_PE', 0) else "Bearish"
        
        delta_exp_ce = row.get('Delta_CE', 0) * row.get('openInterest_CE', 0)
        delta_exp_pe = row.get('Delta_PE', 0) * row.get('openInterest_PE', 0)
        gamma_exp_ce = row.get('Gamma_CE', 0) * row.get('openInterest_CE', 0)
        gamma_exp_pe = row.get('Gamma_PE', 0) * row.get('openInterest_PE', 0)
        
        row_data["DeltaExp_Bias"] = "Bullish" if abs(delta_exp_ce) > abs(delta_exp_pe) else "Bearish"
        row_data["GammaExp_Bias"] = "Bullish" if gamma_exp_ce > gamma_exp_pe else "Bearish"
        row_data["DVP_Bias"] = delta_volume_bias(
            row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
            row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
            row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
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
    total_score = sum([r['Score'] for r in results])
    
    st.success(f"📢 TRADE {'CALL' if best['Score'] > 0 else 'PUT'} | Momentum: {'STRONG' if abs(best['Score']) >= 4 else 'MODERATE'} | Move: {best['FakeReal'].upper()} | Suggested: {best['Scalp/Moment'].upper()}")
    
    # Display total score prominently
    st.subheader(f"📊 Total Bias Score (ATM ±2 Strikes): {total_score}")
    
    st.subheader("📊 Complete Bias Analysis (ATM ±2 Strikes)")
    
    cols = ["Strike", "Zone", "OI_Bias", "ChgOI_Bias", "Volume_Bias", "Delta_Bias", 
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
    
    # New table: Call vs Put values for all biases
    st.subheader("📈 Call vs Put Detailed Analysis (ATM ±2 Strikes)")
    
    detailed_results = []
    for _, row in df.iterrows():
        strike = row['strikePrice']
        zone = 'ATM' if strike == atm_strike else 'ITM' if strike < atm_strike else 'OTM'
        
        detailed_row = {
            "Strike": strike,
            "Zone": zone,
            "LTP_CE": row.get('lastPrice_CE', 0),
            "LTP_PE": row.get('lastPrice_PE', 0),
            "OI_CE": f"{round(row.get('openInterest_CE', 0)/1e6, 2)}M",
            "OI_PE": f"{round(row.get('openInterest_PE', 0)/1e6, 2)}M",
            "ChgOI_CE": f"{int(row.get('changeinOpenInterest_CE', 0)/1000)}K",
            "ChgOI_PE": f"{int(row.get('changeinOpenInterest_PE', 0)/1000)}K",
            "Volume_CE": row.get('totalTradedVolume_CE', 0),
            "Volume_PE": row.get('totalTradedVolume_PE', 0),
            "Delta_CE": row.get('Delta_CE', 0),
            "Delta_PE": row.get('Delta_PE', 0),
            "Gamma_CE": row.get('Gamma_CE', 0),
            "Gamma_PE": row.get('Gamma_PE', 0),
            "Vega_CE": row.get('Vega_CE', 0),
            "Vega_PE": row.get('Vega_PE', 0),
            "Theta_CE": row.get('Theta_CE', 0),
            "Theta_PE": row.get('Theta_PE', 0),
            "IV_CE": round(row.get('impliedVolatility_CE', 0), 2),
            "IV_PE": round(row.get('impliedVolatility_PE', 0), 2),
            "BidQty_CE": row.get('bidQty_CE', 0),
            "AskQty_CE": row.get('askQty_CE', 0),
            "BidQty_PE": row.get('bidQty_PE', 0),
            "AskQty_PE": row.get('askQty_PE', 0),
        }
        detailed_results.append(detailed_row)
    
    detailed_df = pd.DataFrame(detailed_results)
    
    st.dataframe(detailed_df, use_container_width=True, height=400)
    
    # Check for Telegram alerts
    atm_data = [r for r in results if r['Zone'] == 'ATM']
    
    if atm_data:
        atm = atm_data[0]
        
        # Get ATM LTP prices from original dataframe
        atm_row = df[df['strikePrice'] == atm_strike].iloc[0]
        ce_ltp = atm_row['lastPrice_CE']
        pe_ltp = atm_row['lastPrice_PE']
        
        # Check if all biases align
        bias_list = [atm['ChgOI_Bias'], atm['Volume_Bias'], atm['Delta_Bias'], 
                     atm['Gamma_Bias'], atm['AskBid_Bias'], atm['IV_Bias']]
        
        all_bullish = all(b == 'Bullish' for b in bias_list)
        all_bearish = all(b == 'Bearish' for b in bias_list)
        
        # Bull: score > 10, Bear: score < -10
        if (all_bullish and total_score > 10) or (all_bearish and total_score < -10):
            signal = "BULLISH" if all_bullish else "BEARISH"
            option_type = "CALL (CE)" if all_bullish else "PUT (PE)"
            ltp_price = ce_ltp if all_bullish else pe_ltp
            alert_key = f"{atm_strike}_{signal}_{total_score}"
            
            if alert_key not in st.session_state.last_alert:
                message = f"""
🚨 <b>NIFTY TRADING ALERT</b> 🚨

<b>Signal:</b> {signal}
<b>BUY:</b> {option_type}

<b>Spot Price:</b> ₹{underlying:,.2f}
<b>ATM Strike:</b> {atm_strike}
<b>LTP Price:</b> ₹{ltp_price}

<b>Total Score (ATM ±2):</b> {total_score}
<b>Verdict:</b> {atm['Verdict']}

<b>Change in OI Analysis (ATM ±2):</b>
- Total CALL Chg OI: {int(total_chg_oi_ce/1000):,}K
- Total PUT Chg OI: {int(total_chg_oi_pe/1000):,}K
- Difference: {int(chg_oi_diff/1000):,}K ({chg_oi_diff_pct:+.2f}%)
- Chg OI Bias: {chg_oi_bias}

<b>Change in OI Analysis (ATM ±10):</b>
- Total CALL Chg OI: {int(total_chg_oi_ce_wide/1000):,}K
- Total PUT Chg OI: {int(total_chg_oi_pe_wide/1000):,}K
- Difference: {int(chg_oi_diff_wide/1000):,}K ({chg_oi_diff_pct_wide:+.2f}%)
- Chg OI Bias: {chg_oi_bias_wide}

<b>PCR Analysis (ATM ±2):</b>
- PCR (OI): {pcr_oi:.3f}
- PCR (Volume): {pcr_volume:.3f}
- PCR (Chg OI): {pcr_chg_oi:.3f}

<b>PCR Analysis (ATM ±10):</b>
- PCR (OI): {pcr_oi_wide:.3f}
- PCR (Volume): {pcr_volume_wide:.3f}
- PCR (Chg OI): {pcr_chg_oi_wide:.3f}

<b>ATM Bias Analysis:</b>
- ChgOI: {atm['ChgOI_Bias']}
- Volume: {atm['Volume_Bias']}
- Delta: {atm['Delta_Bias']}
- Gamma: {atm['Gamma_Bias']}
- AskBid: {atm['AskBid_Bias']}
- IV: {atm['IV_Bias']}

<b>Trade Suggestion:</b> {atm['Scalp/Moment']}
<b>Entry:</b> {atm['Operator Entry']}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                send_telegram_alert(message)
                st.session_state.last_alert[alert_key] = time.time()
                st.success(f"📲 Telegram Alert Sent: {signal} - BUY {option_type} @ ₹{ltp_price}")
    
except Exception as e:
    st.error(f"Error: {str(e)}")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
