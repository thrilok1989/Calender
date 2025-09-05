import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone
import io
import os
import json

# === Dhan API Configuration ===
try:
    DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
except Exception:
    DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

# === Supabase Configuration ===
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "") 
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
except Exception:
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Initialize Supabase client only if credentials are provided
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("Connected to Supabase")
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
        supabase_client = None
else:
    st.info("Supabase not configured. Add SUPABASE_URL and SUPABASE_KEY to secrets.toml or environment variables to enable data storage.")

# === Streamlit Config ===
st.set_page_config(page_title="Reliance Options Analyzer", layout="wide")
st_autorefresh(interval=23000, key="datarefresh")  # Refresh every 23 seconds for rate limit

# Initialize session state for price data
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])

# Initialize session state for enhanced features
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

if 'call_log_book' not in st.session_state:
    st.session_state.call_log_book = []

if 'export_data' not in st.session_state:
    st.session_state.export_data = False

if 'support_zone' not in st.session_state:
    st.session_state.support_zone = (None, None)

if 'resistance_zone' not in st.session_state:
    st.session_state.resistance_zone = (None, None)

# Initialize PCR-related session state
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 1.2
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.7
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

# === Telegram Config ===
try:
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
except Exception:
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# === Instrument Mapping ===
# RELIANCE underlying instrument ID for Dhan API (NSE Equity)
RELIANCE_UNDERLYING_SCRIP = 2885  # Reliance Industries Ltd underlying instrument ID
RELIANCE_UNDERLYING_SEG = "NSE_EQ"  # NSE Equity segment

# === Dhan API Functions ===
def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str):
    """
    Get option chain data from Dhan API
    """
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    url = "https://api.dhan.co/v2/optionchain"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg,
        "Expiry": expiry
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan option chain: {e}")
        return None

def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str):
    """
    Get expiry list from Dhan API
    """
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan expiry list: {e}")
        return None

# === Supabase Data Management Functions ===
def store_price_data(price):
    """Store price data in Supabase"""
    if not supabase_client:
        return
        
    try:
        data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "price": price,
            "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
        }
        supabase_client.table("reliance_price_history").insert(data).execute()
    except Exception as e:
        st.error(f"Error storing price data: {e}")

def store_trade_log(trade_data):
    """Store trade log entry in Supabase"""
    if not supabase_client:
        return
        
    try:
        # Add timestamp if not present
        if 'Time' not in trade_data:
            trade_data['Time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S")
        
        # Prepare data for Supabase
        supabase_trade_data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "strike": trade_data.get("Strike", 0),
            "option_type": trade_data.get("Type", ""),
            "entry_price": trade_data.get("LTP", 0),
            "target_price": trade_data.get("Target", 0),
            "stop_loss": trade_data.get("SL", 0),
            "pcr": trade_data.get("PCR", 0),
            "pcr_signal": trade_data.get("PCR_Signal", ""),
            "target_hit": trade_data.get("TargetHit", False),
            "sl_hit": trade_data.get("SLHit", False),
            "exit_price": trade_data.get("Exit_Price", None),
            "exit_time": trade_data.get("Exit_Time", None),
            "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
        }
        
        supabase_client.table("reliance_trade_log").insert(supabase_trade_data).execute()
    except Exception as e:
        st.error(f"Error storing trade log: {e}")

def get_trade_log():
    """Get trade log from Supabase"""
    if not supabase_client:
        return []
        
    try:
        response = supabase_client.table("reliance_trade_log") \
            .select("*") \
            .order("timestamp", desc=True) \
            .execute()
        
        if response.data:
            return response.data
        else:
            return []
    except Exception as e:
        st.error(f"Error retrieving trade log: {e}")
        return []

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram credentials not configured")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("Telegram message failed.")
    except Exception as e:
        st.error(f"Telegram error: {e}")

def delete_all_history():
    """Delete all historical data from Supabase"""
    if not supabase_client:
        st.error("Supabase not configured")
        return False
        
    try:
        # Delete all price history
        supabase_client.table("reliance_price_history").delete().neq('id', 0).execute()
        
        # Delete all trade logs
        supabase_client.table("reliance_trade_log").delete().neq('id', 0).execute()
        
        # Clear session state
        if 'price_data' in st.session_state:
            st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])
        if 'pcr_history' in st.session_state:
            st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])
        if 'trade_log' in st.session_state:
            st.session_state.trade_log = []
        if 'call_log_book' in st.session_state:
            st.session_state.call_log_book = []
            
        return True
    except Exception as e:
        st.error(f"Error deleting history: {e}")
        return False

# === Calculation and Analysis Functions ===
def calculate_greeks(option_type, S, K, T, r, sigma):
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
        return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)
    except:
        return 0, 0, 0, 0, 0

def final_verdict(score):
    if score >= 4:
        return "Strong Bullish"
    elif score >= 2:
        return "Bullish"
    elif score <= -4:
        return "Strong Bearish"
    elif score <= -2:
        return "Bearish"
    else:
        return "Neutral"

def delta_volume_bias(price, volume, chg_oi):
    if price > 0 and volume > 0 and chg_oi > 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0:
        return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0:
        return "Bearish"
    else:
        return "Neutral"

def calculate_bid_ask_pressure(call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty):
    """
    Calculate bid/ask pressure - Reliance stock specific thresholds
    """
    pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
    
    # Determine bias based on pressure value - adjusted for individual stock
    if pressure > 300:  # Lower threshold for individual stock
        bias = "Bullish"
    elif pressure < -300:
        bias = "Bearish"
    else:
        bias = "Neutral"
    
    return pressure, bias

# Weights for bias scoring
weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
    "PressureBias": 1,
}

def determine_level(row):
    ce_oi = row.get('openInterest_CE', 0)
    pe_oi = row.get('openInterest_PE', 0)

    if pe_oi > 1.12 * ce_oi:
        return "Support"
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    else:
        return "Neutral"

def is_in_zone(spot, strike, level):
    if level == "Support":
        return strike - 20 <= spot <= strike + 20  # Tighter range for individual stock
    elif level == "Resistance":
        return strike - 20 <= spot <= strike + 20  # Tighter range for individual stock
    return False

# === Display Functions ===
def color_pressure(val):
    if val > 300:  # Adjusted for Reliance stock
        return 'background-color: #90EE90; color: black'  # Light green for bullish
    elif val < -300:
        return 'background-color: #FFB6C1; color: black'  # Light red for bearish
    else:
        return 'background-color: #FFFFE0; color: black'   # Light yellow for neutral

def color_pcr(val):
    if val > st.session_state.pcr_threshold_bull:
        return 'background-color: #90EE90; color: black'
    elif val < st.session_state.pcr_threshold_bear:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

# === Main Analysis Function ===
def analyze():
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("00:00", "%H:%M").time()
        market_end = datetime.strptime("15:40", "%H:%M").time()

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("Market Closed (Mon-Fri 9:00-15:40)")
            return

        # Get expiry list from Dhan API
        expiry_data = get_dhan_expiry_list(RELIANCE_UNDERLYING_SCRIP, RELIANCE_UNDERLYING_SEG)
        if not expiry_data or 'data' not in expiry_data:
            st.error("Failed to get expiry list from Dhan API")
            return
        
        expiry_dates = expiry_data['data']
        if not expiry_dates:
            st.error("No expiry dates available")
            return
        
        expiry = expiry_dates[0]  # Use nearest expiry
        
        # Get option chain from Dhan API
        option_chain_data = get_dhan_option_chain(RELIANCE_UNDERLYING_SCRIP, RELIANCE_UNDERLYING_SEG, expiry)
        if not option_chain_data or 'data' not in option_chain_data:
            st.error("Failed to get option chain from Dhan API")
            return
        
        data = option_chain_data['data']
        underlying = data['last_price']
        
        # Store price data in Supabase
        store_price_data(underlying)

        # Process option chain data
        oc_data = data['oc']
        
        # Convert to DataFrame format
        calls, puts = [], []
        for strike, strike_data in oc_data.items():
            if 'ce' in strike_data:
                ce_data = strike_data['ce']
                ce_data['strikePrice'] = float(strike)
                ce_data['expiryDate'] = expiry
                calls.append(ce_data)
            
            if 'pe' in strike_data:
                pe_data = strike_data['pe']
                pe_data['strikePrice'] = float(strike)
                pe_data['expiryDate'] = expiry
                puts.append(pe_data)
        
        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        
        # Merge call and put data
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
        
        # Rename columns to match standard format
        column_mapping = {
            'last_price': 'lastPrice',
            'oi': 'openInterest',
            'previous_close_price': 'previousClose',
            'previous_oi': 'previousOpenInterest',
            'previous_volume': 'previousVolume',
            'top_ask_price': 'askPrice',
            'top_ask_quantity': 'askQty',
            'top_bid_price': 'bidPrice',
            'top_bid_quantity': 'bidQty',
            'volume': 'totalTradedVolume'
        }
        
        for old_col, new_col in column_mapping.items():
            if f"{old_col}_CE" in df.columns:
                df.rename(columns={f"{old_col}_CE": f"{new_col}_CE"}, inplace=True)
            if f"{old_col}_PE" in df.columns:
                df.rename(columns={f"{old_col}_PE": f"{new_col}_PE"}, inplace=True)
        
        # Calculate change in open interest
        df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
        df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
        
        # Add missing columns with default values
        for col in ['impliedVolatility_CE', 'impliedVolatility_PE']:
            if col not in df.columns:
                df[col] = 0
        
        # Calculate time to expiry
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone("Asia/Kolkata"))
        T = max((expiry_date - now).days, 1) / 365
        r = 0.06

        # Calculate Greeks for calls and puts
        for idx, row in df.iterrows():
            strike = row['strikePrice']
            
            # Calculate Greeks for CE
            try:
                if 'impliedVolatility_CE' in row and row['impliedVolatility_CE'] > 0:
                    greeks = calculate_greeks('CE', underlying, strike, T, r, row['impliedVolatility_CE'] / 100)
                else:
                    greeks = calculate_greeks('CE', underlying, strike, T, r, 0.20)  # 20% default IV for individual stock
            except:
                greeks = (0, 0, 0, 0, 0)
            
            df.at[idx, 'Delta_CE'], df.at[idx, 'Gamma_CE'], df.at[idx, 'Vega_CE'], df.at[idx, 'Theta_CE'], df.at[idx, 'Rho_CE'] = greeks
            
            # Calculate Greeks for PE
            try:
                if 'impliedVolatility_PE' in row and row['impliedVolatility_PE'] > 0:
                    greeks = calculate_greeks('PE', underlying, strike, T, r, row['impliedVolatility_PE'] / 100)
                else:
                    greeks = calculate_greeks('PE', underlying, strike, T, r, 0.20)  # 20% default IV for individual stock
            except:
                greeks = (0, 0, 0, 0, 0)
            
            df.at[idx, 'Delta_PE'], df.at[idx, 'Gamma_PE'], df.at[idx, 'Vega_PE'], df.at[idx, 'Theta_PE'], df.at[idx, 'Rho_PE'] = greeks

        # Analysis logic - Reliance specific adjustments
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        # Reliance strikes are usually in multiples of 25/50, smaller range needed
        strike_range = 200  # Smaller range for individual stock (±200 points)
        df = df[df['strikePrice'].between(atm_strike - strike_range, atm_strike + strike_range)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        # Open Interest Change Comparison
        total_ce_change = df['changeinOpenInterest_CE'].sum() / 100000
        total_pe_change = df['changeinOpenInterest_PE'].sum() / 100000
        
        st.markdown("# Reliance Industries Options Analyzer")
        st.markdown("## Reliance Open Interest Change (in Lakhs)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CALL ΔOI", 
                     f"{total_ce_change:+.1f}L",
                     delta_color="inverse")
            
        with col2:
            st.metric("PUT ΔOI", 
                     f"{total_pe_change:+.1f}L",
                     delta_color="normal")

        # Bias calculation and scoring
        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            # Reliance: Check strikes within ±100 points of ATM
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue

            # Add bid/ask pressure calculation
            bid_ask_pressure, pressure_bias = calculate_bid_ask_pressure(
                row.get('bidQty_CE', 0), 
                row.get('askQty_CE', 0),                                 
                row.get('bidQty_PE', 0), 
                row.get('askQty_PE', 0)
            )
            
            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "Bearish",
                "Volume_Bias": "Bullish" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "Bearish",
                "Gamma_Bias": "Bullish" if row.get('Gamma_CE', 0) < row.get('Gamma_PE', 0) else "Bearish",
                "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish",
                "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish",
                "IV_Bias": "Bullish" if row.get('impliedVolatility_CE', 0) > row.get('impliedVolatility_PE', 0) else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                    row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                    row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
                ),
                "BidAskPressure": bid_ask_pressure,
                "PressureBias": pressure_bias
            }

            # Calculate score based on bias
            for k in row_data:
                if "_Bias" in k:
            # Calculate score based on bias
            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)

            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)
        
        # PCR CALCULATION
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 
                'changeinOpenInterest_CE', 'changeinOpenInterest_PE']],
            left_on='Strike',
            right_on='strikePrice',
            how='left'
        )

        # Calculate PCR
        df_summary['PCR'] = np.where(
            df_summary['openInterest_CE'] == 0,
            0,
            df_summary['openInterest_PE'] / df_summary['openInterest_CE']
        )

        df_summary['PCR'] = df_summary['PCR'].round(2)
        df_summary['PCR_Signal'] = np.where(
            df_summary['PCR'] > st.session_state.pcr_threshold_bull,
            "Bullish",
            np.where(
                df_summary['PCR'] < st.session_state.pcr_threshold_bear,
                "Bearish",
                "Neutral"
            )
        )

        # Style the dataframe
        styled_df = df_summary.style.applymap(color_pcr, subset=['PCR']).applymap(color_pressure, subset=['BidAskPressure'])
        df_summary = df_summary.drop(columns=['strikePrice'])

        # Calculate market view
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"

        # Main Display
        st.markdown(f"### Reliance Stock Price: ₹{underlying:,.2f}")
        st.success(f"Market View: **{market_view}** | Bias Score: {total_score}")

        with st.expander("Reliance Option Chain Summary"):
            st.info(f"""
            **Reliance PCR Interpretation:**
            - >{st.session_state.pcr_threshold_bull} = Strong Put Activity (Bullish Signal)
            - <{st.session_state.pcr_threshold_bear} = Strong Call Activity (Bearish Signal)
            - PCR Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'}
            - Lot Size: 250 contracts
            - Strike Range: ±200 points from ATM
            - Refresh Rate: 23 seconds (Rate Limited)
            - Default IV: 20% (Individual Stock)
            """)
            
            st.dataframe(styled_df)

        # Enhanced Features Display
        st.markdown("---")
        st.markdown("## Enhanced Features")
        
        # PCR Configuration
        st.markdown("### PCR Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pcr_threshold_bull = st.number_input(
                "Bullish PCR Threshold (>)", 
                min_value=1.0, max_value=5.0, 
                value=st.session_state.pcr_threshold_bull, 
                step=0.1
            )
        with col2:
            st.session_state.pcr_threshold_bear = st.number_input(
                "Bearish PCR Threshold (<)", 
                min_value=0.1, max_value=1.0, 
                value=st.session_state.pcr_threshold_bear, 
                step=0.1
            )
        with col3:
            st.session_state.use_pcr_filter = st.checkbox(
                "Enable PCR Filtering", 
                value=st.session_state.use_pcr_filter
            )

        # Data Management Section
        st.markdown("---")
        st.markdown("### Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("Export functionality available in full version")
        
        with col2:
            if st.button("Delete All History", type="secondary", use_container_width=True):
                if st.session_state.get('confirm_delete', False):
                    if delete_all_history():
                        st.success("All Reliance history deleted successfully!")
                        send_telegram_message("Reliance: All historical data deleted")
                        st.session_state.confirm_delete = False
                        st.rerun()
                    else:
                        st.error("Failed to delete history")
                        st.session_state.confirm_delete = False
                else:
                    st.session_state.confirm_delete = True
                    st.warning("Click again to confirm deletion of ALL historical data")
        
        # Reset confirmation if user doesn't click again
        if st.session_state.get('confirm_delete', False):
            if st.button("Cancel Deletion"):
                st.session_state.confirm_delete = False
                st.info("Deletion cancelled")

    except Exception as e:
        st.error(f"Error: {e}")
        send_telegram_message(f"Reliance Error: {str(e)}")

# Main Function Call
if __name__ == "__main__":
    analyze()