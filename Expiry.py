import os
import math
import json
import time
import pathlib
import streamlit as st
import threading
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests
from pytz import timezone
import random

# ===================== CONFIG =====================
SYMBOL = os.getenv("NSE_SYMBOL", "NIFTY").upper()  # "NIFTY" or "BANKNIFTY"
EXCEL_FILE = os.getenv("EXCEL_FILE", f"expiry_day_{SYMBOL.lower()}_analysis.xlsx")
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", f"snapshots_{SYMBOL.lower()}")
pathlib.Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

ZONE_WIDTH = int(os.getenv("ZONE_WIDTH", 10))
GAMMA_BREACH = int(os.getenv("GAMMA_BREACH", 30))
FAR_FROM_MAXPAIN = int(os.getenv("FAR_FROM_MAXPAIN", 100))
PCR_BULL = float(os.getenv("PCR_BULL", 1.2))
PCR_BEAR = float(os.getenv("PCR_BEAR", 0.9))
TOP_N_DELTA_OI = int(os.getenv("TOP_N_DELTA_OI", 5))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", 20))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))

# Timezone configuration
IST = timezone('Asia/Kolkata')

# Telegram Config (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BASE = "https://www.nseindia.com"
OC_URL = f"{BASE}/api/option-chain-indices?symbol={SYMBOL}"

# List of realistic user agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

# ===================== TIME UTILITIES =====================
def get_ist_time():
    """Get current time in IST timezone"""
    return datetime.now(IST)

def format_ist_time(dt=None, format_str="%Y-%m-%d %H:%M:%S"):
    """Format datetime object to IST time string"""
    if dt is None:
        dt = get_ist_time()
    elif dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.strftime(format_str)

def get_naive_ist_time():
    """Get current IST time as naive datetime (for session state storage)"""
    return get_ist_time().replace(tzinfo=None)

# ===================== TELEGRAM FUNCTIONS =====================
def send_telegram_message(message):
    """Send message via Telegram bot (optional feature)"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    try:
        try:
            import telegram
        except ImportError:
            st.warning("Telegram package not installed. Install with: pip install python-telegram-bot")
            return False
            
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        return True
    except Exception as e:
        st.error(f"Failed to send Telegram message: {e}")
        return False

# ===================== TIME VALIDATION =====================
def is_valid_trading_time():
    """Check if current time is within trading hours (Mon-Fri, 9:00-15:30 IST)"""
    now = get_ist_time()
    
    if now.weekday() > 4:
        return False
    
    current_time = now.time()
    start_time = datetime.strptime("09:00", "%H:%M").time()
    end_time = datetime.strptime("15:30", "%H:%M").time()
    
    return start_time <= current_time <= end_time

# ===================== IMPROVED DATA FETCH WITH COOKIE HANDLING =====================
def get_random_user_agent():
    """Get a random user agent from the list"""
    return random.choice(USER_AGENTS)

def _nse_session() -> requests.Session:
    """Create a session with proper cookies and headers"""
    s = requests.Session()
    
    # Set headers with random user agent
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": f"{BASE}/",
        "Origin": BASE,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    s.headers.update(headers)
    
    # First visit to get cookies
    try:
        # Visit main page to get initial cookies
        s.get(BASE, timeout=HTTP_TIMEOUT)
        time.sleep(1)
        
        # Visit market data page to get additional cookies
        s.get(f"{BASE}/market-data/live-equity-market", timeout=HTTP_TIMEOUT)
        time.sleep(0.5)
        
        # Visit options page
        s.get(f"{BASE}/market-data/equity-derivatives-watch", timeout=HTTP_TIMEOUT)
        time.sleep(0.5)
        
    except Exception as e:
        st.warning(f"Cookie setup warning: {e}")
    
    return s

def fetch_option_chain_with_retry(max_retries=MAX_RETRIES):
    """Fetch option chain data with robust retry logic"""
    session = None
    
    for attempt in range(max_retries):
        try:
            # Create new session for each attempt or if previous failed
            if session is None:
                session = _nse_session()
            
            # Add delay with jitter between retries
            if attempt > 0:
                delay = 2 + (attempt * 1) + random.uniform(0, 0.5)
                time.sleep(delay)
                # Rotate user agent on retry
                session.headers.update({"User-Agent": get_random_user_agent()})
            
            # Make the request
            response = session.get(OC_URL, timeout=HTTP_TIMEOUT)
            
            # Check for 401/403 errors
            if response.status_code in [401, 403]:
                st.warning(f"Access denied (attempt {attempt + 1}/{max_retries}). Creating new session...")
                session = None  # Force new session on next attempt
                continue
                
            response.raise_for_status()
            
            # Validate response content
            content = response.content.strip()
            if not content:
                st.warning(f"Empty response (attempt {attempt + 1}/{max_retries})")
                continue
                
            data = response.json()
            
            # Validate response structure
            if not isinstance(data, dict) or 'records' not in data:
                st.warning(f"Invalid response structure (attempt {attempt + 1}/{max_retries})")
                continue
                
            return data
            
        except requests.exceptions.JSONDecodeError:
            st.warning(f"Invalid JSON response (attempt {attempt + 1}/{max_retries})")
            session = None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [401, 403]:
                st.warning(f"HTTP error {e.response.status_code} (attempt {attempt + 1}/{max_retries})")
                session = None
            else:
                st.warning(f"HTTP error: {e} (attempt {attempt + 1}/{max_retries})")
            
        except requests.exceptions.RequestException as e:
            st.warning(f"Network error: {e} (attempt {attempt + 1}/{max_retries})")
            session = None
            
        except Exception as e:
            st.warning(f"Unexpected error: {e} (attempt {attempt + 1}/{max_retries})")
            session = None
    
    # Fallback: try with a direct request without session (simpler approach)
    try:
        st.info("Trying fallback method...")
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "application/json",
            "Referer": f"{BASE}/"
        }
        response = requests.get(OC_URL, headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Failed to fetch option chain data after {max_retries} attempts: {e}")

def fetch_option_chain() -> dict:
    """Main function to fetch option chain data"""
    return fetch_option_chain_with_retry()

# ===================== ALTERNATIVE DATA SOURCE (FALLBACK) =====================
def fetch_from_alternative_source():
    """Fallback method using alternative data source"""
    try:
        # Try using a proxy or alternative endpoint
        alt_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={SYMBOL}"
        
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "application/json",
            "Referer": f"{BASE}/",
            "X-Requested-With": "XMLHttpRequest"
        }
        
        # Try with a fresh session
        session = requests.Session()
        session.headers.update(headers)
        
        # Get cookies first
        session.get(BASE, timeout=HTTP_TIMEOUT)
        time.sleep(1)
        
        response = session.get(alt_url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        st.error(f"Alternative source also failed: {e}")
        return None

# ===================== PARSE & CORE METRICS =====================
def parse_chain(raw: dict) -> Tuple[pd.DataFrame, float, str, str]:
    if not raw or 'records' not in raw:
        raise ValueError("Invalid or empty response from NSE")
        
    records = raw.get("records", {})
    filtered = raw.get("filtered", {})

    spot = float(records.get("underlyingValue") or filtered.get("underlyingValue") or 0.0)
    expiries = records.get("expiryDates") or filtered.get("expiryDates") or []
    expiry = expiries[0] if expiries else ""
    rows = []
    
    for item in records.get("data", []):
        sp = item.get("strikePrice")
        if sp is None:
            continue
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        rows.append({
            "strikePrice": float(sp),
            "CE_OI": int(ce.get("openInterest", 0) or 0),
            "CE_ChgOI": int(ce.get("changeinOpenInterest", 0) or 0),
            "CE_LTP": float(ce.get("lastPrice", float("nan"))),
            "PE_OI": int(pe.get("openInterest", 0) or 0),
            "PE_ChgOI": int(pe.get("changeinOpenInterest", 0) or 0),
            "PE_LTP": float(pe.get("lastPrice", float("nan"))),
        })
    
    df = pd.DataFrame(rows).sort_values("strikePrice").reset_index(drop=True)
    ts_str = format_ist_time()
    return df, spot, expiry, ts_str

# ... [rest of the functions remain the same until main()] ...

# ===================== MAIN ANALYSIS FUNCTION =====================
def run_analysis():
    try:
        if not is_valid_trading_time():
            st.warning("Outside trading hours (Mon-Fri, 9:00-15:30 IST). Analysis paused.")
            return None

        try:
            raw = fetch_option_chain()
        except Exception as e:
            st.error(f"Primary fetch failed: {e}")
            st.info("Trying alternative source...")
            raw = fetch_from_alternative_source()
            if not raw:
                raise Exception("All data sources failed")

        df, spot, expiry, ts = parse_chain(raw)

        # core metrics
        pcr = calculate_pcr(df)
        max_pain = calculate_max_pain(df)
        support, resistance = sr_levels_from_oi(df)
        support, resistance, sr_bias = pcr_weighted_sr(support, resistance, pcr)
        atm_k, atm_ce, atm_pe, atm_total = atm_straddle(df, spot)
        gamma = gamma_status(spot, support, resistance)
        mp_strength_label, mp_strength_pct, mp_strength_score = max_pain_strength(df, spot, max_pain)
        
        # ΔOI vs latest snapshot
        prev_df = load_latest_snapshot()
        delta_df = compute_delta_oi(df, prev_df)
        top_ce, top_pe = top_delta_oi(delta_df)
        
        # merge ΔOI into chain for excel
        chain_aug = pd.merge(df, delta_df, on='strikePrice', how='left')
        chain_aug.insert(0, 'Timestamp', ts)
        
        # max pain shift vs ~1 hour ago
        hour_df, hour_ts = load_snapshot_older_than(55)
        if hour_df is not None:
            mp_hour = calculate_max_pain(hour_df)
            mp_shift = int(max_pain - mp_hour)
        else:
            mp_shift = None
        
        # Bias table (ATM ±3) using chain_aug (has delta columns)
        bias_df, bias_total = bias_table_atm_band(chain_aug, spot, max_pain)
        bias_df.insert(0, 'Timestamp', ts)
        
        # overall score and signal
        total_score = overall_score(pcr, mp_strength_score, gamma, bias_total, spot, max_pain)
        signal = make_trade_signal(spot, pcr, mp_shift, gamma, support, resistance, bias_total)
        
        # Overview row
        overview = pd.DataFrame([{
            "Timestamp": ts,
            "Symbol": SYMBOL,
            "Expiry": expiry,
            "Spot": spot,
            "PCR": pcr,
            "MaxPain": max_pain,
            "MaxPainShift_vs_1h": mp_shift if mp_shift is not None else "NA",
            "MaxPainStrength": mp_strength_label,
            "MaxPainStrength%": round(mp_strength_pct, 2),
            "MaxPainStrengthScore": mp_strength_score,
            "Support": support,
            "Resistance": resistance,
            "SR_Bias": sr_bias,
            "ZoneWidth": ZONE_WIDTH,
            "Gamma": gamma,
            "ATM_Strike": atm_k,
            "ATM_CE_LTP": atm_ce,
            "ATM_PE_LTP": atm_pe,
            "ATM_Straddle": atm_total,
            "BiasTotal(ATM±3)": bias_total,
            "Score": total_score,
            "Signal": signal.get("Signal"),
            "Side": signal.get("Side"),
            "EntryZone": signal.get("EntryZone"),
            "SL": signal.get("SL"),
            "Target": signal.get("Target"),
            "Reason": signal.get("Reason"),
        }])
        
        # Signals log row (for append)
        signals_log = overview[[
            "Timestamp", "Symbol", "Spot", "PCR", "MaxPain", "MaxPainShift_vs_1h",
            "MaxPainStrength", "MaxPainStrength%", "MaxPainStrengthScore",
            "Support", "Resistance", "Gamma", "ATM_Strike", "ATM_Straddle",
            "BiasTotal(ATM±3)", "Score", "Signal", "Side", "EntryZone", "SL", "Target", "Reason"
        ]].copy()
        
        # Save snapshot and excel
        snap_path = save_snapshot(df, ts)
        try:
            append_to_excel(overview, chain_aug, top_ce, top_pe, bias_df, signals_log)
        except Exception as e:
            st.warning(f"Excel save failed: {e}")
        
        # Send Telegram notification if there's a signal
        if signal["Signal"] != "WAIT":
            telegram_msg = f"{SYMBOL} Signal: {signal['Signal']}\nSpot: {spot}, PCR: {pcr}\nEntry: {signal['EntryZone']}\nReason: {signal['Reason']}\nTime: {ts} IST"
            send_telegram_message(telegram_msg)
        
        return {
            "overview": overview,
            "chain_aug": chain_aug,
            "top_ce": top_ce,
            "top_pe": top_pe,
            "bias_df": bias_df,
            "signals_log": signals_log,
            "snap_path": snap_path,
            "timestamp": ts
        }
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        # Provide sample data for demonstration
        if st.checkbox("Show sample data for demonstration"):
            return get_sample_data()
    
    return None

def get_sample_data():
    """Return sample data for demonstration when live data fails"""
    ts = format_ist_time()
    return {
        "overview": pd.DataFrame([{
            "Timestamp": ts,
            "Symbol": SYMBOL,
            "Expiry": "28-DEC-2023",
            "Spot": 21500.50,
            "PCR": 1.15,
            "MaxPain": 21500,
            "MaxPainShift_vs_1h": 0,
            "MaxPainStrength": "strong",
            "MaxPainStrength%": 28.5,
            "MaxPainStrengthScore": 2,
            "Support": 21400,
            "Resistance": 21600,
            "SR_Bias": "Support Bias (Bullish)",
            "ZoneWidth": ZONE_WIDTH,
            "Gamma": "Neutral",
            "ATM_Strike": 21500,
            "ATM_CE_LTP": 150.25,
            "ATM_PE_LTP": 145.75,
            "ATM_Straddle": 296.0,
            "BiasTotal(ATM±3)": 3,
            "Score": 65,
            "Signal": "WAIT",
            "Side": "-",
            "EntryZone": "-",
            "SL": "-",
            "Target": "-",
            "Reason": "No alignment",
        }]),
        "timestamp": ts,
        "snap_path": "sample_data.csv"
    }

# ===================== STREAMLIT UI =====================
def main():
    st.set_page_config(
        page_title=f"{SYMBOL} Expiry Day Analytics",
        page_icon="📈",
        layout="wide"
    )
    
    st.title(f"{SYMBOL} Expiry Day Analytics Dashboard")
    st.write(f"Auto-refreshing every 2 minutes | Mon-Fri 9:00-15:30 IST | Current IST Time: {format_ist_time()}")
    
    # Initialize session state
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Auto-refresh logic
    refresh_interval = 120
    current_time = get_naive_ist_time()
    
    if st.session_state.last_update is None or (current_time - st.session_state.last_update).seconds >= refresh_interval:
        with st.spinner("Fetching latest data..."):
            st.session_state.data = run_analysis()
            st.session_state.last_update = current_time
    
    # Display data if available
    if st.session_state.data:
        data = st.session_state.data
        overview = data["overview"]
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot", overview["Spot"].iloc[0])
            st.metric("PCR", overview["PCR"].iloc[0])
        with col2:
            st.metric("Max Pain", overview["MaxPain"].iloc[0])
            st.metric("Max Pain Strength", overview["MaxPainStrength"].iloc[0])
        with col3:
            st.metric("Support", overview["Support"].iloc[0])
            st.metric("Resistance", overview["Resistance"].iloc[0])
        with col4:
            st.metric("Signal", overview["Signal"].iloc[0])
            st.metric("Overall Score", overview["Score"].iloc[0])
        
        # Display detailed sections
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Bias Analysis", "Top ΔOI", "Chain Data"])
        
        with tab1:
            st.subheader("Overview")
            st.dataframe(overview)
            
        with tab2:
            st.subheader("Bias Analysis (ATM ±3 Strikes)")
            if "bias_df" in data:
                st.dataframe(data["bias_df"])
                st.metric("Total Bias Score", overview["BiasTotal(ATM±3)"].iloc[0])
            else:
                st.info("Bias analysis data not available")
        
        st.write(f"Last updated: {data['timestamp']} IST")
    
    # Manual refresh button
    if st.button("Refresh Now"):
        with st.spinner("Refreshing data..."):
            st.session_state.data = run_analysis()
            st.session_state.last_update = get_naive_ist_time()
        st.rerun()

if __name__ == "__main__":
    main()
