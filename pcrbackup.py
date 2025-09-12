import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
from datetime import datetime
from pytz import timezone
import time
import os

# Configuration
st.set_page_config(page_title="Dhan Options Bot", layout="wide")
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", os.getenv("DHAN_CLIENT_ID", ""))
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", os.getenv("DHAN_ACCESS_TOKEN", ""))
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))

if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
    st.error("Configure DHAN credentials in .streamlit/secrets.toml")
    st.stop()

# Session state
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = {}

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
    return now.weekday() < 5 and datetime.strptime("09:15", "%H:%M").time() <= now.time() <= datetime.strptime("15:30", "%H:%M").time()

def analyze_atm(df, spot):
    atm = min(df['strikePrice'], key=lambda x: abs(x - spot))
    row = df[df['strikePrice'] == atm].iloc[0]
    
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
    pe_1_5x_ce = abs(pe_oi_chg) > 1.5 * abs(ce_oi_chg) if ce_oi_chg != 0 else abs(pe_oi_chg) > 1000
    ce_1_5x_pe = abs(ce_oi_chg) > 1.5 * abs(pe_oi_chg) if pe_oi_chg != 0 else abs(ce_oi_chg) > 1000
    
    return {
        'atm': atm, 'spot': spot, 'biases': biases, 'all_bullish': all_bullish, 'all_bearish': all_bearish,
        'pe_1_5x_ce': pe_1_5x_ce, 'ce_1_5x_pe': ce_1_5x_pe, 'ce_ltp': row.get('lastPrice_CE', 0),
        'pe_ltp': row.get('lastPrice_PE', 0), 'ce_oi_chg': ce_oi_chg, 'pe_oi_chg': pe_oi_chg
    }

def check_signal(data):
    current_time = time.time()
    cooldown = 300
    
    if data['pe_1_5x_ce'] and data['all_bullish'] and current_time - st.session_state.last_signal_time.get('bullish', 0) > cooldown:
        return "BULLISH"
    elif data['ce_1_5x_pe'] and data['all_bearish'] and current_time - st.session_state.last_signal_time.get('bearish', 0) > cooldown:
        return "BEARISH"
    return None

def main():
    st.title("Dhan Options Trading Bot")
    
    if not is_market_open():
        st.warning("Market Closed")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Config")
        st.write(f"Client: {DHAN_CLIENT_ID[:8]}...")
        st.write(f"Telegram: {'✅' if TELEGRAM_BOT_TOKEN else '❌'}")
        
        if TELEGRAM_BOT_TOKEN and st.button("Test Telegram"):
            if send_telegram("🧪 Test from Dhan Bot"):
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
    signal = check_signal(analysis)
    
    # Display
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("NIFTY", f"{spot:.2f}")
    with col2: st.metric("ATM Strike", analysis['atm'])
    with col3: st.metric("Time", datetime.now().strftime("%H:%M:%S"))
    
    # Biases
    st.subheader("ATM Biases")
    bias_df = pd.DataFrame([{"Bias": k, "Value": v, "Status": "✅" if v == "Bullish" else "❌"} 
                           for k, v in analysis['biases'].items()])
    st.dataframe(bias_df, hide_index=True)
    
    # OI Analysis
    col1, col2 = st.columns(2)
    with col1:
        ratio = abs(analysis['pe_oi_chg'] / analysis['ce_oi_chg']) if analysis['ce_oi_chg'] != 0 else float('inf')
        st.metric("PE/CE OI Ratio", f"{ratio:.2f}x")
        if analysis['pe_1_5x_ce']: st.success("✅ PE > 1.5x CE")
        if analysis['all_bullish']: st.success("✅ All Bullish")
    
    with col2:
        ratio = abs(analysis['ce_oi_chg'] / analysis['pe_oi_chg']) if analysis['pe_oi_chg'] != 0 else float('inf')
        st.metric("CE/PE OI Ratio", f"{ratio:.2f}x")
        if analysis['ce_1_5x_pe']: st.error("✅ CE > 1.5x PE")
        if analysis['all_bearish']: st.error("✅ All Bearish")
    
    # Signal handling
    if signal == "BULLISH":
        st.session_state.last_signal_time['bullish'] = time.time()
        st.success("🚨 BULLISH SIGNAL!")
        
        stop_loss = analysis['ce_ltp'] * 0.8
        msg = f"""
🟢 BULLISH SIGNAL - BUY CALL

Strike: {analysis['atm']}
Entry: ₹{analysis['ce_ltp']:.2f}
Stop Loss: ₹{stop_loss:.2f} (20%)

All Biases Bullish + PE OI Change > 1.5x CE OI Change
Time: {datetime.now().strftime("%H:%M:%S")}
        """
        st.code(msg)
        send_telegram(msg.strip())
    
    elif signal == "BEARISH":
        st.session_state.last_signal_time['bearish'] = time.time()
        st.error("🚨 BEARISH SIGNAL!")
        
        stop_loss = analysis['pe_ltp'] * 0.8
        msg = f"""
🔴 BEARISH SIGNAL - BUY PUT

Strike: {analysis['atm']}
Entry: ₹{analysis['pe_ltp']:.2f}
Stop Loss: ₹{stop_loss:.2f} (20%)

All Biases Bearish + CE OI Change > 1.5x PE OI Change
Time: {datetime.now().strftime("%H:%M:%S")}
        """
        st.code(msg)
        send_telegram(msg.strip())

if __name__ == "__main__":
    main()
