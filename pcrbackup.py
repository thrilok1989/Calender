import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
import time
import os
import numpy as np

# Configuration
st.set_page_config(page_title="RSI Dual Signal Dhan Bot", layout="wide")
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", os.getenv("DHAN_CLIENT_ID", ""))
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", os.getenv("DHAN_ACCESS_TOKEN", ""))
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))

if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
    st.error("Configure DHAN credentials in .streamlit/secrets.toml")
    st.stop()

# Session state
if 'cooldowns' not in st.session_state:
    st.session_state.cooldowns = {}

# Auto-refresh
st_autorefresh(interval=30000, key="refresh")

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        return requests.post(url, data=payload).status_code == 200
    except:
        return False

def dhan_api(endpoint, payload):
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(f"https://api.dhan.co/v2{endpoint}", headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def is_market_open():
    now = datetime.now(timezone("Asia/Kolkata"))
    return now.weekday() < 5 and datetime.strptime("09:15", "%H:%M").time() <= now.time() <= datetime.strptime("18:30", "%H:%M").time()

def calculate_rsi(prices, period=7):
    """Calculate RSI with given period"""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate for remaining periods
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)

def get_rsi_data():
    """Fetch intraday data and calculate RSI-7"""
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        # Get data from market open to now
        from_date = now.strftime("%Y-%m-%d 09:15:00")
        to_date = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Try NIFTY 50 Index - Security ID 13
        payload = {
            "securityId": "13",
            "exchangeSegment": "IDX_I", 
            "instrument": "INDEX",
            "interval": "1",
            "fromDate": from_date,
            "toDate": to_date
        }
        
        data = dhan_api("/charts/intraday", payload)
        
        # Debug: Print data length for troubleshooting
        if data and 'close' in data:
            st.write(f"Debug: Got {len(data['close'])} data points")  # Remove this after testing
            
            if len(data['close']) >= 14:  # Need minimum 14 points for RSI-7
                prices = data['close']
                rsi = calculate_rsi(prices, 7)
                
                if rsi is not None:
                    if rsi > 70:
                        level = "Overbought"
                    elif rsi < 30:
                        level = "Oversold"
                    else:
                        level = "Neutral"
                    return rsi, level
            else:
                st.write(f"Debug: Insufficient data - need 14, got {len(data['close'])}")
        else:
            st.write("Debug: No data received from API")
            
        return None, "N/A"
    except Exception as e:
        st.write(f"Debug RSI Error: {e}")  # Remove this after testing
        return None, "N/A"

def analyze_atm(df, spot):
    atm = min(df['strikePrice'], key=lambda x: abs(x - spot))
    row = df[df['strikePrice'] == atm].iloc[0]
    
    ce_oi = row.get('openInterest_CE', 0)
    pe_oi = row.get('openInterest_PE', 0)
    ce_oi_chg = row.get('changeinOpenInterest_CE', 0)
    pe_oi_chg = row.get('changeinOpenInterest_PE', 0)
    
    # Bias calculations
    biases = {
        'ChgOI': "Bullish" if pe_oi_chg > ce_oi_chg else "Bearish",
        'Volume': "Bullish" if row.get('totalTradedVolume_PE', 0) > row.get('totalTradedVolume_CE', 0) else "Bearish",
        'AskQty': "Bullish" if row.get('askQty_CE', 0) > row.get('askQty_PE', 0) else "Bearish",
        'BidQty': "Bullish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bearish",
        'Pressure': "Bullish" if (row.get('bidQty_CE', 0) - row.get('askQty_CE', 0) + row.get('askQty_PE', 0) - row.get('bidQty_PE', 0)) > 500 else "Bearish"
    }
    
    all_bullish = all(b == "Bullish" for b in biases.values())
    all_bearish = all(b == "Bearish" for b in biases.values())
    
    # Primary signal conditions (OI Change ratios)
    pe_chg_1_5x_ce = abs(pe_oi_chg) > 1.5 * abs(ce_oi_chg) if ce_oi_chg != 0 else abs(pe_oi_chg) > 1000
    ce_chg_1_5x_pe = abs(ce_oi_chg) > 1.5 * abs(pe_oi_chg) if pe_oi_chg != 0 else abs(ce_oi_chg) > 1000
    
    # Secondary signal conditions (Absolute OI ratios)
    pe_oi_1_3x_ce = pe_oi > 1.3 * ce_oi if ce_oi > 0 else False
    ce_oi_1_3x_pe = ce_oi > 1.3 * pe_oi if pe_oi > 0 else False
    
    # Get RSI data
    rsi_value, rsi_level = get_rsi_data()
    
    return {
        'atm': atm, 'spot': spot, 'biases': biases, 'all_bullish': all_bullish, 'all_bearish': all_bearish,
        'pe_chg_1_5x_ce': pe_chg_1_5x_ce, 'ce_chg_1_5x_pe': ce_chg_1_5x_pe,
        'pe_oi_1_3x_ce': pe_oi_1_3x_ce, 'ce_oi_1_3x_pe': ce_oi_1_3x_pe,
        'ce_ltp': row.get('lastPrice_CE', 0), 'pe_ltp': row.get('lastPrice_PE', 0),
        'ce_oi_chg': ce_oi_chg, 'pe_oi_chg': pe_oi_chg, 'ce_oi': ce_oi, 'pe_oi': pe_oi,
        'rsi_value': rsi_value, 'rsi_level': rsi_level
    }

def check_signals(data):
    current_time = time.time()
    cooldown = 300
    signals = []
    
    # Primary signals (OI Change based)
    if data['pe_chg_1_5x_ce'] and data['all_bullish'] and current_time - st.session_state.cooldowns.get('p_bull', 0) > cooldown:
        signals.append(('PRIMARY_BULLISH', 'p_bull'))
    
    if data['ce_chg_1_5x_pe'] and data['all_bearish'] and current_time - st.session_state.cooldowns.get('p_bear', 0) > cooldown:
        signals.append(('PRIMARY_BEARISH', 'p_bear'))
    
    # Secondary signals (Absolute OI based)
    if data['pe_oi_1_3x_ce'] and data['all_bullish'] and current_time - st.session_state.cooldowns.get('s_bull', 0) > cooldown:
        signals.append(('SECONDARY_BULLISH', 's_bull'))
    
    if data['ce_oi_1_3x_pe'] and data['all_bearish'] and current_time - st.session_state.cooldowns.get('s_bear', 0) > cooldown:
        signals.append(('SECONDARY_BEARISH', 's_bear'))
    
    return signals

def main():
    st.title("RSI Dual Signal Dhan Options Bot")
    
    if not is_market_open():
        st.warning("Market Closed")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Config")
        st.write(f"Client: {DHAN_CLIENT_ID[:8]}...")
        st.write(f"Telegram: {'✅' if TELEGRAM_BOT_TOKEN else '❌'}")
        
        st.subheader("RSI Settings")
        st.write("Period: 7")
        st.write("Oversold: < 30")
        st.write("Overbought: > 70")
        
        st.subheader("Signals")
        st.write("**Primary:** OI Change > 1.5x")
        st.write("**Secondary:** Absolute OI > 1.3x")
        
        if TELEGRAM_BOT_TOKEN and st.button("Test Telegram", use_container_width=True):
            test_msg = f"🧪 Test from RSI Dual Signal Bot\nTime: {datetime.now().strftime('%H:%M:%S')}"
            if send_telegram(test_msg):
                st.success("✅ Working!")
            else:
                st.error("❌ Failed!")
    
    # Get data
    expiry_data = dhan_api("/optionchain/expirylist", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"})
    if not expiry_data:
        st.error("Failed to fetch expiry data")
        return
    
    expiry = expiry_data['data'][0]
    oc_data = dhan_api("/optionchain", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry})
    if not oc_data:
        st.error("Failed to fetch option chain")
        return
    
    # Process data
    spot = oc_data['data']['last_price']
    oc = oc_data['data']['oc']
    
    df_data = []
    for strike, data in oc.items():
        if 'ce' in data and 'pe' in data:
            row = {'strikePrice': float(strike)}
            for suffix, option_data in [('_CE', data['ce']), ('_PE', data['pe'])]:
                row[f'lastPrice{suffix}'] = option_data.get('last_price', 0)
                row[f'openInterest{suffix}'] = option_data.get('oi', 0)
                row[f'previousOpenInterest{suffix}'] = option_data.get('previous_oi', 0)
                row[f'totalTradedVolume{suffix}'] = option_data.get('volume', 0)
                row[f'bidQty{suffix}'] = option_data.get('top_bid_quantity', 0)
                row[f'askQty{suffix}'] = option_data.get('top_ask_quantity', 0)
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
    df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
    
    # Analysis
    analysis = analyze_atm(df, spot)
    signals = check_signals(analysis)
    
    # Display
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("NIFTY", f"{spot:.2f}")
    with col2: st.metric("ATM Strike", analysis['atm'])
    with col3: 
        rsi_color = "🔴" if analysis['rsi_level'] == "Overbought" else "🟢" if analysis['rsi_level'] == "Oversold" else "🟡"
        st.metric("RSI-7", f"{rsi_color} {analysis['rsi_value'] or 'N/A'}")
    with col4: st.metric("Time", datetime.now().strftime("%H:%M:%S"))
    
    # RSI Level Display
    if analysis['rsi_value']:
        if analysis['rsi_level'] == "Overbought":
            st.error(f"RSI-7: {analysis['rsi_value']} (Overbought)")
        elif analysis['rsi_level'] == "Oversold":
            st.success(f"RSI-7: {analysis['rsi_value']} (Oversold)")
        else:
            st.info(f"RSI-7: {analysis['rsi_value']} (Neutral)")
    
    # Biases
    st.subheader("ATM Biases")
    bias_df = pd.DataFrame([{"Bias": k, "Value": v, "Status": "✅" if v == "Bullish" else "❌"} 
                           for k, v in analysis['biases'].items()])
    st.dataframe(bias_df, hide_index=True)
    
    # Signal Conditions
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primary Signals")
        chg_ratio = abs(analysis['pe_oi_chg'] / analysis['ce_oi_chg']) if analysis['ce_oi_chg'] != 0 else float('inf')
        st.metric("PE/CE OI Change", f"{chg_ratio:.2f}x")
        if analysis['pe_chg_1_5x_ce'] and analysis['all_bullish']: st.success("✅ Primary CALL Ready")
        if analysis['ce_chg_1_5x_pe'] and analysis['all_bearish']: st.error("✅ Primary PUT Ready")
    
    with col2:
        st.subheader("Secondary Signals") 
        oi_ratio = analysis['pe_oi'] / analysis['ce_oi'] if analysis['ce_oi'] > 0 else float('inf')
        st.metric("PE/CE OI Absolute", f"{oi_ratio:.2f}x")
        if analysis['pe_oi_1_3x_ce'] and analysis['all_bullish']: st.success("✅ Secondary CALL Ready")
        if analysis['ce_oi_1_3x_pe'] and analysis['all_bearish']: st.error("✅ Secondary PUT Ready")
    
    # Process signals
    for signal_type, cooldown_key in signals:
        st.session_state.cooldowns[cooldown_key] = time.time()
        
        rsi_text = f"RSI-7: {analysis['rsi_value']} ({analysis['rsi_level']})" if analysis['rsi_value'] else "RSI-7: N/A"
        
        if 'BULLISH' in signal_type:
            st.success(f"🚨 {signal_type.replace('_', ' ')} SIGNAL!")
            stop_loss = analysis['ce_ltp'] * 0.8
            signal_prefix = "PRIMARY" if "PRIMARY" in signal_type else "SECONDARY"
            
            msg = f"""
🟢 <b>{signal_prefix} BULLISH SIGNAL - BUY CALL</b>

Strike: {analysis['atm']}
Entry: ₹{analysis['ce_ltp']:.2f}
Stop Loss: ₹{stop_loss:.2f} (20%)
{rsi_text}

Condition: {"PE OI Change > 1.5x CE" if "PRIMARY" in signal_type else "PUT OI > 1.3x CALL OI"}
All Biases: Bullish
Time: {datetime.now().strftime("%H:%M:%S")}
            """
            
        else:  # BEARISH
            st.error(f"🚨 {signal_type.replace('_', ' ')} SIGNAL!")
            stop_loss = analysis['pe_ltp'] * 0.8
            signal_prefix = "PRIMARY" if "PRIMARY" in signal_type else "SECONDARY"
            
            msg = f"""
🔴 <b>{signal_prefix} BEARISH SIGNAL - BUY PUT</b>

Strike: {analysis['atm']}
Entry: ₹{analysis['pe_ltp']:.2f}
Stop Loss: ₹{stop_loss:.2f} (20%)
{rsi_text}

Condition: {"CE OI Change > 1.5x PE" if "PRIMARY" in signal_type else "CALL OI > 1.3x PUT OI"}
All Biases: Bearish
Time: {datetime.now().strftime("%H:%M:%S")}
            """
        
        st.code(msg)
        send_telegram(msg.strip())

if __name__ == "__main__":
    main()