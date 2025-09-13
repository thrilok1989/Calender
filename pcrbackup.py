import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
from datetime import datetime
from pytz import timezone
import time
import os

# Config
st.set_page_config(page_title="RSI Dual Signal Bot", layout="wide")
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
if 'rsi_prices' not in st.session_state:
    st.session_state.rsi_prices = []

# Auto refresh
st_autorefresh(interval=30000, key="refresh")

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        return requests.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'}).status_code == 200
    except:
        return False

def dhan_api(endpoint, payload):
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    try:
        response = requests.post(f"https://api.dhan.co/v2{endpoint}", headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def is_market_open():
    now = datetime.now(timezone("Asia/Kolkata"))
    return now.weekday() < 6 and datetime.strptime("09:15", "%H:%M").time() <= now.time() <= datetime.strptime("15:30", "%H:%M").time()

def calculate_rsi(prices, period=7):
    """Calculate RSI using Wilder's smoothing (TradingView method)"""
    if len(prices) < period + 1:
        return None
    
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [max(0, change) for change in changes]
    losses = [max(0, -change) for change in changes]
    
    # Initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Wilder's smoothing for remaining periods
    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def get_current_ltp():
    """Get current NIFTY LTP from market quote"""
    try:
        payload = {"IDX_I": [13]}
        data = dhan_api("/marketfeed/ltp", payload)
        if data and 'data' in data and 'IDX_I' in data['data'] and '13' in data['data']['IDX_I']:
            return data['data']['IDX_I']['13']['last_price']
        return None
    except:
        return None

def update_rsi_prices():
    """Update price history and calculate RSI"""
    current_ltp = get_current_ltp()
    if current_ltp is None:
        return None, "N/A"
    
    # Add current price to history
    if 'rsi_prices' not in st.session_state:
        st.session_state.rsi_prices = []
    
    # Only add if price is different from last price (avoid duplicates)
    if not st.session_state.rsi_prices or st.session_state.rsi_prices[-1] != current_ltp:
        st.session_state.rsi_prices.append(current_ltp)
        
        # Keep only last 50 prices for efficiency
        if len(st.session_state.rsi_prices) > 50:
            st.session_state.rsi_prices = st.session_state.rsi_prices[-50:]
    
    # Calculate RSI if we have enough data points
    if len(st.session_state.rsi_prices) >= 8:
        rsi = calculate_rsi(st.session_state.rsi_prices, 7)
        if rsi is not None:
            if rsi > 70:
                level = "Overbought"
            elif rsi < 30:
                level = "Oversold"
            else:
                level = "Neutral"
            return rsi, level
    
    return None, "N/A"

def get_rsi():
    """Get RSI using live market data"""
    return update_rsi_prices()

def analyze_atm(df, spot):
    atm = min(df['strikePrice'], key=lambda x: abs(x - spot))
    row = df[df['strikePrice'] == atm].iloc[0]
    
    ce_oi = row.get('openInterest_CE', 0)
    pe_oi = row.get('openInterest_PE', 0)
    ce_oi_chg = row.get('changeinOpenInterest_CE', 0)
    pe_oi_chg = row.get('changeinOpenInterest_PE', 0)
    
    # Biases
    biases = {
        'ChgOI': "Bullish" if pe_oi_chg > ce_oi_chg else "Bearish",
        'Volume': "Bullish" if row.get('totalTradedVolume_PE', 0) > row.get('totalTradedVolume_CE', 0) else "Bearish",
        'AskQty': "Bullish" if row.get('askQty_CE', 0) > row.get('askQty_PE', 0) else "Bearish",
        'BidQty': "Bullish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bearish",
        'Pressure': "Bullish" if (row.get('bidQty_CE', 0) - row.get('askQty_CE', 0) + row.get('askQty_PE', 0) - row.get('bidQty_PE', 0)) > 500 else "Bearish"
    }
    
    all_bullish = all(b == "Bullish" for b in biases.values())
    all_bearish = all(b == "Bearish" for b in biases.values())
    
    # Primary conditions (OI Change ratios)
    pe_chg_1_5x_ce = abs(pe_oi_chg) > 1.5 * abs(ce_oi_chg) if ce_oi_chg != 0 else abs(pe_oi_chg) > 1000
    ce_chg_1_5x_pe = abs(ce_oi_chg) > 1.5 * abs(pe_oi_chg) if pe_oi_chg != 0 else abs(ce_oi_chg) > 1000
    
    # Secondary conditions (Absolute OI ratios)
    pe_oi_1_3x_ce = pe_oi > 1.3 * ce_oi if ce_oi > 0 else False
    ce_oi_1_3x_pe = ce_oi > 1.3 * pe_oi if pe_oi > 0 else False
    
    # RSI
    rsi_value, rsi_level = get_rsi()
    
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
    
    # Primary signals
    if data['pe_chg_1_5x_ce'] and data['all_bullish'] and current_time - st.session_state.cooldowns.get('p_bull', 0) > cooldown:
        signals.append(('PRIMARY_BULLISH', 'p_bull'))
    
    if data['ce_chg_1_5x_pe'] and data['all_bearish'] and current_time - st.session_state.cooldowns.get('p_bear', 0) > cooldown:
        signals.append(('PRIMARY_BEARISH', 'p_bear'))
    
    # Secondary signals
    if data['pe_oi_1_3x_ce'] and data['all_bullish'] and current_time - st.session_state.cooldowns.get('s_bull', 0) > cooldown:
        signals.append(('SECONDARY_BULLISH', 's_bull'))
    
    if data['ce_oi_1_3x_pe'] and data['all_bearish'] and current_time - st.session_state.cooldowns.get('s_bear', 0) > cooldown:
        signals.append(('SECONDARY_BEARISH', 's_bear'))
    
    return signals

def main():
    st.title("RSI Dual Signal Dhan Bot")
    
    if not is_market_open():
        st.warning("Market Closed")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Config")
        st.write(f"Client: {DHAN_CLIENT_ID[:8]}...")
        st.write(f"Telegram: {'✅' if TELEGRAM_BOT_TOKEN else '❌'}")
        
        st.subheader("RSI-7")
        st.write("Method: Wilder's (TradingView)")
        st.write("Oversold: < 30, Overbought: > 70")
        
        st.subheader("Signals")
        st.write("**Primary:** OI Change > 1.5x + All Biases")
        st.write("**Secondary:** OI > 1.3x + All Biases")
        
        if TELEGRAM_BOT_TOKEN and st.button("Test Telegram", use_container_width=True):
            if send_telegram(f"🧪 Test from RSI Bot\nTime: {datetime.now().strftime('%H:%M:%S')}"):
                st.success("✅ Working!")
            else:
                st.error("❌ Failed!")
    
    # Get data
    expiry_data = dhan_api("/optionchain/expirylist", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"})
    if not expiry_data:
        st.error("Failed to fetch expiry")
        return
    
    oc_data = dhan_api("/optionchain", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry_data['data'][0]})
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
    with col2: st.metric("ATM", analysis['atm'])
    with col3:
        if analysis['rsi_value']:
            rsi_color = "🔴" if analysis['rsi_level'] == "Overbought" else "🟢" if analysis['rsi_level'] == "Oversold" else "🟡"
            st.metric("RSI-7", f"{rsi_color} {analysis['rsi_value']}")
        else:
            st.metric("RSI-7", "N/A")
    with col4: st.metric("Time", datetime.now().strftime("%H:%M:%S"))
    
    # RSI Status
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
    
    # Conditions
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primary")
        chg_ratio = abs(analysis['pe_oi_chg'] / analysis['ce_oi_chg']) if analysis['ce_oi_chg'] != 0 else float('inf')
        st.metric("PE/CE Change", f"{chg_ratio:.2f}x")
        if analysis['pe_chg_1_5x_ce'] and analysis['all_bullish']: st.success("✅ CALL Ready")
        if analysis['ce_chg_1_5x_pe'] and analysis['all_bearish']: st.error("✅ PUT Ready")
    
    with col2:
        st.subheader("Secondary")
        oi_ratio = analysis['pe_oi'] / analysis['ce_oi'] if analysis['ce_oi'] > 0 else float('inf')
        st.metric("PE/CE OI", f"{oi_ratio:.2f}x")
        if analysis['pe_oi_1_3x_ce'] and analysis['all_bullish']: st.success("✅ CALL Ready")
        if analysis['ce_oi_1_3x_pe'] and analysis['all_bearish']: st.error("✅ PUT Ready")
    
    # Process signals
    for signal_type, cooldown_key in signals:
        st.session_state.cooldowns[cooldown_key] = time.time()
        
        rsi_text = f"RSI-7: {analysis['rsi_value']} ({analysis['rsi_level']})" if analysis['rsi_value'] else "RSI-7: N/A"
        
        if 'BULLISH' in signal_type:
            st.success(f"🚨 {signal_type.replace('_', ' ')} SIGNAL!")
            stop_loss = analysis['ce_ltp'] * 0.8
            signal_prefix = "PRIMARY" if "PRIMARY" in signal_type else "SECONDARY"
            
            msg = f"""
🟢 <b>{signal_prefix} BULLISH - BUY CALL</b>

Strike: {analysis['atm']}
Entry: ₹{analysis['ce_ltp']:.2f}
Stop Loss: ₹{stop_loss:.2f} (20%)
{rsi_text}

Condition: {"PE OI Change > 1.5x CE" if "PRIMARY" in signal_type else "PUT OI > 1.3x CALL OI"}
All Biases: Bullish

Time: {datetime.now().strftime("%H:%M:%S")}
            """
            
        else:
            st.error(f"🚨 {signal_type.replace('_', ' ')} SIGNAL!")
            stop_loss = analysis['pe_ltp'] * 0.8
            signal_prefix = "PRIMARY" if "PRIMARY" in signal_type else "SECONDARY"
            
            msg = f"""
🔴 <b>{signal_prefix} BEARISH - BUY PUT</b>

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