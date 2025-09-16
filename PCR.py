import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
import pytz

# Page config
st.set_page_config(page_title="Nifty Options SuperTrend", layout="wide")

# CSS
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

class DhanAPI:
    def __init__(self):
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'access-token': st.secrets["DHAN_ACCESS_TOKEN"],
            'client-id': st.secrets["DHAN_CLIENT_ID"],
            'Content-Type': 'application/json'
        }
    
    def get_option_chain(self, underlying_scrip, underlying_seg, expiry):
        url = f"{self.base_url}/optionchain"
        payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg, "Expiry": expiry}
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_expiry_list(self, underlying_scrip, underlying_seg):
        url = f"{self.base_url}/optionchain/expirylist"
        payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg}
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None

class TelegramBot:
    def __init__(self):
        self.bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
    
    def send_message(self, message):
        if not self.bot_token or not self.chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        try:
            response = requests.post(url, data=data, timeout=5)
            return response.status_code == 200
        except:
            return False

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    if len(high) < period:
        return [], []
    
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    atr = []
    
    # Calculate ATR
    tr = []
    for i in range(1, len(high)):
        tr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr.append(tr_val)
    
    if len(tr) >= period:
        atr.append(sum(tr[:period]) / period)
        for i in range(period, len(tr)):
            atr.append((atr[-1] * (period - 1) + tr[i]) / period)
    
    # Calculate SuperTrend
    supertrend = []
    trend_direction = []
    upper_band = []
    lower_band = []
    
    for i in range(len(hl2)):
        if i < len(atr):
            ub = hl2[i] + multiplier * atr[i]
            lb = hl2[i] - multiplier * atr[i]
        else:
            ub = hl2[i] + multiplier * (atr[-1] if atr else 0)
            lb = hl2[i] - multiplier * (atr[-1] if atr else 0)
        
        if i == 0:
            upper_band.append(ub)
            lower_band.append(lb)
            supertrend.append(ub)
            trend_direction.append(1)
        else:
            ub = ub if ub < upper_band[-1] or close[i-1] > upper_band[-1] else upper_band[-1]
            lb = lb if lb > lower_band[-1] or close[i-1] < lower_band[-1] else lower_band[-1]
            
            upper_band.append(ub)
            lower_band.append(lb)
            
            if trend_direction[-1] == 1 and close[i] <= lb:
                trend_direction.append(-1)
            elif trend_direction[-1] == -1 and close[i] >= ub:
                trend_direction.append(1)
            else:
                trend_direction.append(trend_direction[-1])
            
            supertrend.append(lb if trend_direction[-1] == 1 else ub)
    
    return supertrend, trend_direction

def create_candlestick_chart(data, strike_price, option_type):
    if not data or len(data) < 4:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=f"{strike_price} {option_type}",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # SuperTrend
    if len(df) >= 10:
        supertrend, trend_direction = calculate_supertrend(
            df['high'].tolist(), df['low'].tolist(), df['close'].tolist()
        )
        
        # Split into up and down
        st_up = [st if td == 1 else None for st, td in zip(supertrend, trend_direction)]
        st_down = [st if td == -1 else None for st, td in zip(supertrend, trend_direction)]
        
        fig.add_trace(go.Scatter(
            x=df.index[-len(st_up):],
            y=st_up,
            mode='lines',
            name='SuperTrend Up',
            line=dict(color='#00ff88', width=2),
            connectgaps=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index[-len(st_down):],
            y=st_down,
            mode='lines',
            name='SuperTrend Down',
            line=dict(color='#ff4444', width=2),
            connectgaps=False
        ))
        
        # Check for trend change (arrow formation)
        if len(trend_direction) >= 2:
            current_trend = trend_direction[-1]
            previous_trend = trend_direction[-2]
            
            if current_trend != previous_trend:
                return fig, current_trend  # Return trend change
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title="Time Points",
        yaxis_title="Price (₹)",
        showlegend=True
    )
    
    return fig, None

def simulate_ohlc_from_ltp(ltp, volatility=0.02):
    """Create OHLC data from LTP"""
    noise = np.random.normal(0, volatility * ltp, 4)
    high = ltp + abs(noise[0])
    low = ltp - abs(noise[1])
    open_price = ltp + noise[2]
    close = ltp
    
    # Ensure OHLC relationships
    high = max(high, open_price, close, low)
    low = min(low, open_price, close, high)
    
    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'time': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
    }

def main():
    st.title("Nifty Options SuperTrend Trader")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = {}
    if 'last_trends' not in st.session_state:
        st.session_state.last_trends = {}
    
    api = DhanAPI()
    telegram = TelegramBot()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Strike selection
    strikes = st.sidebar.multiselect(
        "Select Strikes (ATM ±2)",
        ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2'],
        default=['ATM-1', 'ATM', 'ATM+1']
    )
    
    option_type = st.sidebar.selectbox("Option Type", ['CE', 'PE'])
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox("Refresh Interval", [5, 10, 15, 30], index=1)
    
    # Get expiry dates
    expiry_data = api.get_expiry_list(13, "IDX_I")
    if expiry_data and 'data' in expiry_data:
        expiry = st.sidebar.selectbox("Expiry", expiry_data['data'])
    else:
        expiry = "2024-10-31"
    
    # Manual refresh
    if st.sidebar.button("Refresh Now"):
        st.rerun()
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    # Get option chain data
    option_chain = api.get_option_chain(13, "IDX_I", expiry)
    
    if option_chain and 'data' in option_chain:
        spot_price = option_chain['data'].get('last_price', 25000)
        atm_strike = round(spot_price / 50) * 50
        oc_data = option_chain['data'].get('oc', {})
        
        with col1:
            st.metric("Nifty Spot", f"₹{spot_price:.2f}")
        with col2:
            st.metric("ATM Strike", atm_strike)
        with col3:
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
            st.metric("IST Time", ist_time)
        
        # Process each selected strike
        for strike_ref in strikes:
            # Calculate actual strike price
            if strike_ref == 'ATM':
                strike_price = atm_strike
            elif strike_ref.startswith('ATM+'):
                offset = int(strike_ref.replace('ATM+', ''))
                strike_price = atm_strike + (offset * 50)
            elif strike_ref.startswith('ATM-'):
                offset = int(strike_ref.replace('ATM-', ''))
                strike_price = atm_strike - (offset * 50)
            
            strike_key = f"{strike_price}.000000"
            data_key = f"{strike_ref}_{option_type}"
            
            # Get option price
            if strike_key in oc_data:
                option_data = oc_data[strike_key].get(option_type.lower(), {})
                ltp = option_data.get('last_price', 0)
                
                if ltp > 0:
                    # Initialize data storage
                    if data_key not in st.session_state.data:
                        st.session_state.data[data_key] = []
                    
                    # Add OHLC data point
                    ohlc_point = simulate_ohlc_from_ltp(ltp)
                    st.session_state.data[data_key].append(ohlc_point)
                    
                    # Keep only last 50 points
                    if len(st.session_state.data[data_key]) > 50:
                        st.session_state.data[data_key] = st.session_state.data[data_key][-50:]
                    
                    # Create chart
                    st.subheader(f"{strike_price} {option_type} - ₹{ltp:.2f}")
                    
                    chart_result = create_candlestick_chart(
                        st.session_state.data[data_key], 
                        strike_price, 
                        option_type
                    )
                    
                    if isinstance(chart_result, tuple):
                        fig, trend_change = chart_result
                        
                        # Check for SuperTrend arrow formation
                        if trend_change is not None:
                            last_trend = st.session_state.last_trends.get(data_key)
                            
                            if last_trend != trend_change:
                                # SuperTrend arrow formed
                                trend_text = "BULLISH ⬆️" if trend_change == 1 else "BEARISH ⬇️"
                                
                                message = f"""
<b>SuperTrend Alert!</b>
Strike: {strike_price} {option_type}
Signal: {trend_text}
Price: ₹{ltp:.2f}
Time: {ist_time} IST
Nifty: ₹{spot_price:.2f}
                                """.strip()
                                
                                # Send Telegram notification
                                if telegram.send_message(message):
                                    st.success(f"Alert sent: {trend_text}")
                                else:
                                    st.warning("Telegram notification failed")
                                
                                # Update last trend
                                st.session_state.last_trends[data_key] = trend_change
                    else:
                        fig = chart_result
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Failed to fetch option chain data")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**Setup Telegram Notifications:**")
    st.code("""
# Add to Streamlit secrets:
DHAN_ACCESS_TOKEN = "your_token"
DHAN_CLIENT_ID = "your_client_id"
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
""")

if __name__ == "__main__":
    main()
