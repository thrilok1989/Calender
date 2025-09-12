import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from pytz import timezone
import os
import math
from scipy.stats import norm

# === Streamlit Configuration ===
st.set_page_config(
    page_title="Dhan Options Trading Monitor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Configuration from Streamlit Secrets ===
try:
    if hasattr(st, 'secrets'):
        DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", os.getenv("DHAN_CLIENT_ID"))
        DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", os.getenv("DHAN_ACCESS_TOKEN"))
        TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN"))
        TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID"))
    else:
        DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
        DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    if not all([DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN]):
        st.error("Please configure DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in Streamlit secrets")
        st.stop()
        
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

NIFTY_UNDERLYING_SCRIP = 13
NIFTY_UNDERLYING_SEG = "IDX_I"
STOP_LOSS_PERCENTAGE = 20

# === Initialize Session State ===
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = {}
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []

# Auto-refresh every 30 seconds
count = st_autorefresh(interval=30000, key="datarefresh")

# === Utility Functions ===
def send_telegram_message(message):
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram credentials not configured - signals will only show on screen")
        return False
        
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Telegram error: {e}")
        return False

def get_dhan_option_chain(underlying_scrip, underlying_seg, expiry):
    """Fetch option chain from Dhan API"""
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
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

def get_dhan_expiry_list(underlying_scrip, underlying_seg):
    """Fetch expiry list from Dhan API"""
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
    except Exception as e:
        st.error(f"Error fetching expiry list: {e}")
        return None

def is_market_open():
    """Check if market is open"""
    now = datetime.now(timezone("Asia/Kolkata"))
    current_day = now.weekday()
    current_time = now.time()
    market_start = datetime.strptime("09:15", "%H:%M").time()
    market_end = datetime.strptime("15:30", "%H:%M").time()
    
    return current_day < 5 and market_start <= current_time <= market_end

def calculate_bid_ask_pressure(call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty):
    """Calculate bid-ask pressure"""
    pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
    if pressure > 500:
        bias = "Bullish"
    elif pressure < -500:
        bias = "Bearish"
    else:
        bias = "Neutral"
    return pressure, bias

def determine_level(ce_oi, pe_oi):
    """Determine if strike is support or resistance"""
    if pe_oi > 1.12 * ce_oi:
        return "Support"
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    else:
        return "Neutral"

def process_option_chain_data(option_chain_data):
    """Process option chain data and create DataFrame"""
    if not option_chain_data or 'data' not in option_chain_data:
        return None, None
    
    data = option_chain_data['data']
    underlying = data['last_price']
    
    # Flatten option chain data
    oc_data = data['oc']
    calls, puts = [], []
    
    for strike, strike_data in oc_data.items():
        if 'ce' in strike_data:
            ce_data = strike_data['ce']
            ce_data['strikePrice'] = float(strike)
            calls.append(ce_data)
        if 'pe' in strike_data:
            pe_data = strike_data['pe']
            pe_data['strikePrice'] = float(strike)
            puts.append(pe_data)
    
    if not calls or not puts:
        return None, underlying
    
    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
    
    # Rename columns
    column_mapping = {
        'last_price': 'lastPrice',
        'oi': 'openInterest',
        'previous_oi': 'previousOpenInterest',
        'top_ask_quantity': 'askQty',
        'top_bid_quantity': 'bidQty',
        'volume': 'totalTradedVolume'
    }
    
    for old_col, new_col in column_mapping.items():
        if f"{old_col}_CE" in df.columns:
            df.rename(columns={f"{old_col}_CE": f"{new_col}_CE"}, inplace=True)
        if f"{old_col}_PE" in df.columns:
            df.rename(columns={f"{old_col}_PE": f"{new_col}_PE"}, inplace=True)
    
    # Calculate change in OI
    df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
    df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
    
    return df, underlying

def analyze_atm_conditions(df, underlying_price):
    """Analyze ATM conditions for trading signals"""
    # Find ATM strike
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying_price))
    atm_row = df[df['strikePrice'] == atm_strike].iloc[0]
    
    # Extract ATM data
    ce_oi = atm_row.get('openInterest_CE', 0)
    pe_oi = atm_row.get('openInterest_PE', 0)
    ce_oi_change = atm_row.get('changeinOpenInterest_CE', 0)
    pe_oi_change = atm_row.get('changeinOpenInterest_PE', 0)
    ce_volume = atm_row.get('totalTradedVolume_CE', 0)
    pe_volume = atm_row.get('totalTradedVolume_PE', 0)
    ce_bid_qty = atm_row.get('bidQty_CE', 0)
    pe_bid_qty = atm_row.get('bidQty_PE', 0)
    ce_ask_qty = atm_row.get('askQty_CE', 0)
    pe_ask_qty = atm_row.get('askQty_PE', 0)
    ce_ltp = atm_row.get('lastPrice_CE', 0)
    pe_ltp = atm_row.get('lastPrice_PE', 0)
    
    # Determine level
    level = determine_level(ce_oi, pe_oi)
    
    # Calculate bid-ask pressure
    pressure, pressure_bias = calculate_bid_ask_pressure(ce_bid_qty, ce_ask_qty, pe_bid_qty, pe_ask_qty)
    
    # Bullish conditions check
    bullish_conditions = {
        'OI Change': pe_oi_change > ce_oi_change,
        'Volume': pe_volume > ce_volume,
        'Bid Qty': pe_bid_qty > ce_bid_qty,
        'Ask Qty': ce_ask_qty > pe_ask_qty,
        'Level': level == "Support",
        'Pressure': pressure_bias == "Bullish"
    }
    
    # Bearish conditions check
    bearish_conditions = {
        'OI Change': ce_oi_change > pe_oi_change,
        'Volume': ce_volume > pe_volume,
        'Bid Qty': ce_bid_qty > pe_bid_qty,
        'Ask Qty': pe_ask_qty > ce_ask_qty,
        'Level': level == "Resistance",
        'Pressure': pressure_bias == "Bearish"
    }
    
    bullish_score = sum(bullish_conditions.values())
    bearish_score = sum(bearish_conditions.values())
    
    return {
        'atm_strike': atm_strike,
        'underlying_price': underlying_price,
        'level': level,
        'ce_ltp': ce_ltp,
        'pe_ltp': pe_ltp,
        'bullish_score': bullish_score,
        'bearish_score': bearish_score,
        'bullish_conditions': bullish_conditions,
        'bearish_conditions': bearish_conditions,
        'pressure': pressure,
        'pressure_bias': pressure_bias,
        'ce_oi_change': ce_oi_change,
        'pe_oi_change': pe_oi_change,
        'ce_volume': ce_volume,
        'pe_volume': pe_volume,
        'ce_oi': ce_oi,
        'pe_oi': pe_oi
    }

def check_signal_conditions(signal_data):
    """Check if signal conditions are met"""
    current_time = time.time()
    signal_cooldown = 300  # 5 minutes
    
    # Check for BULLISH signal
    if (signal_data['bullish_score'] >= 4 and 
        signal_data['level'] == "Support" and
        current_time - st.session_state.last_signal_time.get('bullish', 0) > signal_cooldown):
        return "BULLISH"
    
    # Check for BEARISH signal
    elif (signal_data['bearish_score'] >= 4 and 
          signal_data['level'] == "Resistance" and
          current_time - st.session_state.last_signal_time.get('bearish', 0) > signal_cooldown):
        return "BEARISH"
    
    return None

# === Main App ===
def main():
    st.title("🚀 Dhan Options Trading Monitor")
    
    # Market status
    market_open = is_market_open()
    if market_open:
        st.success("🟢 Market is OPEN")
    else:
        st.warning("🔴 Market is CLOSED")
    
    if not market_open:
        st.info("Market Hours: Monday-Friday, 9:15 AM - 3:30 PM IST")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        st.write(f"**Client ID:** {DHAN_CLIENT_ID}")
        st.write(f"**Telegram:** {'✅ Configured' if TELEGRAM_BOT_TOKEN else '❌ Not configured'}")
        st.write(f"**Stop Loss:** {STOP_LOSS_PERCENTAGE}%")
        st.write(f"**Refresh:** Every 30 seconds")
        
        if st.button("Test Telegram"):
            if send_telegram_message("🧪 Test message from Dhan Options Monitor"):
                st.success("Telegram working!")
            else:
                st.error("Telegram failed!")
    
    # Get expiry list
    with st.spinner("Fetching expiry dates..."):
        expiry_data = get_dhan_expiry_list(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
        if not expiry_data or 'data' not in expiry_data:
            st.error("Failed to fetch expiry dates")
            return
        
        expiry_dates = expiry_data['data']
        if not expiry_dates:
            st.error("No expiry dates available")
            return
    
    expiry = expiry_dates[0]  # Use nearest expiry
    
    # Get option chain data
    with st.spinner("Fetching option chain data..."):
        option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
        df, underlying_price = process_option_chain_data(option_chain_data)
        
        if df is None:
            st.error("Failed to process option chain data")
            return
    
    # Display current info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NIFTY Spot", f"{underlying_price:.2f}")
    with col2:
        st.metric("Expiry", expiry)
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    # Analyze ATM conditions
    signal_data = analyze_atm_conditions(df, underlying_price)
    
    # Display ATM Analysis
    st.header(f"📊 ATM Analysis - Strike {signal_data['atm_strike']}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Level", signal_data['level'])
    with col2:
        st.metric("CALL LTP", f"₹{signal_data['ce_ltp']:.2f}")
    with col3:
        st.metric("PUT LTP", f"₹{signal_data['pe_ltp']:.2f}")
    with col4:
        st.metric("Pressure", f"{signal_data['pressure']:,}")
    
    # Signal Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Bullish Conditions")
        bullish_df = pd.DataFrame([
            {"Condition": k, "Status": "✅" if v else "❌", "Met": v}
            for k, v in signal_data['bullish_conditions'].items()
        ])
        st.dataframe(bullish_df, hide_index=True)
        st.metric("Bullish Score", f"{signal_data['bullish_score']}/6")
    
    with col2:
        st.subheader("🔴 Bearish Conditions")
        bearish_df = pd.DataFrame([
            {"Condition": k, "Status": "✅" if v else "❌", "Met": v}
            for k, v in signal_data['bearish_conditions'].items()
        ])
        st.dataframe(bearish_df, hide_index=True)
        st.metric("Bearish Score", f"{signal_data['bearish_score']}/6")
    
    # Check for signals
    signal_type = check_signal_conditions(signal_data)
    
    if signal_type == "BULLISH":
        st.session_state.last_signal_time['bullish'] = time.time()
        
        atm_strike = signal_data['atm_strike']
        ce_ltp = signal_data['ce_ltp']
        stop_loss = ce_ltp * (100 - STOP_LOSS_PERCENTAGE) / 100
        
        st.success("🚨 BULLISH SIGNAL DETECTED!")
        
        signal_msg = f"""
🟢 BULLISH SIGNAL - BUY CALL

📊 ATM Details:
• Strike: {atm_strike}
• Level: {signal_data['level']}
• NIFTY Spot: {signal_data['underlying_price']:.2f}

💰 Trade Setup:
• BUY {atm_strike} CALL
• Entry Price: ₹{ce_ltp:.2f}
• Stop Loss: ₹{stop_loss:.2f} ({STOP_LOSS_PERCENTAGE}%)

📈 Signal Strength: {signal_data['bullish_score']}/6

📊 Supporting Data:
• PUT OI Change: {signal_data['pe_oi_change']:,}
• CALL OI Change: {signal_data['ce_oi_change']:,}
• PUT Volume: {signal_data['pe_volume']:,}
• CALL Volume: {signal_data['ce_volume']:,}
• Bid-Ask Pressure: {signal_data['pressure']:,} ({signal_data['pressure_bias']})

⏰ Time: {datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        st.code(signal_msg)
        send_telegram_message(signal_msg.strip())
        
        # Add to history
        st.session_state.signal_history.append({
            'time': datetime.now(),
            'type': 'BULLISH',
            'strike': atm_strike,
            'entry': ce_ltp,
            'stop_loss': stop_loss
        })
    
    elif signal_type == "BEARISH":
        st.session_state.last_signal_time['bearish'] = time.time()
        
        atm_strike = signal_data['atm_strike']
        pe_ltp = signal_data['pe_ltp']
        stop_loss = pe_ltp * (100 - STOP_LOSS_PERCENTAGE) / 100
        
        st.error("🚨 BEARISH SIGNAL DETECTED!")
        
        signal_msg = f"""
🔴 BEARISH SIGNAL - BUY PUT

📊 ATM Details:
• Strike: {atm_strike}
• Level: {signal_data['level']}
• NIFTY Spot: {signal_data['underlying_price']:.2f}

💰 Trade Setup:
• BUY {atm_strike} PUT
• Entry Price: ₹{pe_ltp:.2f}
• Stop Loss: ₹{stop_loss:.2f} ({STOP_LOSS_PERCENTAGE}%)

📉 Signal Strength: {signal_data['bearish_score']}/6

📊 Supporting Data:
• CALL OI Change: {signal_data['ce_oi_change']:,}
• PUT OI Change: {signal_data['pe_oi_change']:,}
• CALL Volume: {signal_data['ce_volume']:,}
• PUT Volume: {signal_data['pe_volume']:,}
• Bid-Ask Pressure: {signal_data['pressure']:,} ({signal_data['pressure_bias']})

⏰ Time: {datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        st.code(signal_msg)
        send_telegram_message(signal_msg.strip())
        
        # Add to history
        st.session_state.signal_history.append({
            'time': datetime.now(),
            'type': 'BEARISH',
            'strike': atm_strike,
            'entry': pe_ltp,
            'stop_loss': stop_loss
        })
    
    # Signal History
    if st.session_state.signal_history:
        st.header("📈 Signal History")
        history_df = pd.DataFrame(st.session_state.signal_history)
        st.dataframe(history_df, hide_index=True)
    
    # Raw data display (optional)
    with st.expander("📊 Raw Option Chain Data"):
        display_df = df[['strikePrice', 'lastPrice_CE', 'lastPrice_PE', 
                        'openInterest_CE', 'openInterest_PE', 
                        'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                        'totalTradedVolume_CE', 'totalTradedVolume_PE']]
        st.dataframe(display_df)

if __name__ == "__main__":
    main()
