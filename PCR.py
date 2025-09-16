import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta
import pytz

# Page config
st.set_page_config(page_title="Nifty Options SuperTrend", layout="wide")

# CSS
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
div[data-testid="metric-container"] { background-color: #262730; border: 1px solid #434651; padding: 10px; border-radius: 5px; }
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
        except Exception as e:
            st.error(f"API Error: {e}")
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
    if len(high) < period + 1:
        return [], []
    
    # Calculate True Range
    tr = []
    for i in range(1, len(high)):
        tr_val = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr.append(tr_val)
    
    # Calculate ATR
    atr = []
    if len(tr) >= period:
        atr.append(sum(tr[:period]) / period)
        for i in range(period, len(tr)):
            atr.append((atr[-1] * (period - 1) + tr[i]) / period)
    
    # Calculate SuperTrend
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    supertrend = []
    trend_direction = []
    
    for i in range(len(hl2)):
        if i < len(atr):
            ub = hl2[i] + multiplier * atr[i]
            lb = hl2[i] - multiplier * atr[i]
        else:
            if atr:
                ub = hl2[i] + multiplier * atr[-1]
                lb = hl2[i] - multiplier * atr[-1]
            else:
                ub = hl2[i]
                lb = hl2[i]
        
        if i == 0:
            supertrend.append(ub)
            trend_direction.append(1)
        else:
            prev_ub = supertrend[-1] if trend_direction[-1] == -1 else ub
            prev_lb = supertrend[-1] if trend_direction[-1] == 1 else lb
            
            if ub < prev_ub or close[i-1] > prev_ub:
                final_ub = ub
            else:
                final_ub = prev_ub
                
            if lb > prev_lb or close[i-1] < prev_lb:
                final_lb = lb
            else:
                final_lb = prev_lb
            
            if trend_direction[-1] == 1 and close[i] <= final_lb:
                trend_direction.append(-1)
                supertrend.append(final_ub)
            elif trend_direction[-1] == -1 and close[i] >= final_ub:
                trend_direction.append(1)
                supertrend.append(final_lb)
            else:
                trend_direction.append(trend_direction[-1])
                if trend_direction[-1] == 1:
                    supertrend.append(final_lb)
                else:
                    supertrend.append(final_ub)
    
    return supertrend, trend_direction

def collect_live_data(api, strike_price, option_type, expiry, max_points=100):
    """Collect real-time LTP data points"""
    
    # Initialize session state for data storage
    data_key = f"live_data_{strike_price}_{option_type}_{expiry}"
    
    if data_key not in st.session_state:
        st.session_state[data_key] = []
    
    # Get current option chain data
    option_chain = api.get_option_chain(13, "IDX_I", expiry)
    
    if option_chain and 'data' in option_chain:
        oc_data = option_chain['data'].get('oc', {})
        strike_key = f"{strike_price}.000000"
        
        if strike_key in oc_data:
            option_data = oc_data[strike_key].get(option_type.lower(), {})
            current_ltp = option_data.get('last_price', 0)
            
            if current_ltp > 0:
                ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                
                # Create OHLC from current LTP
                if len(st.session_state[data_key]) == 0:
                    # First data point
                    ohlc_point = {
                        'timestamp': ist_now,
                        'open': current_ltp,
                        'high': current_ltp,
                        'low': current_ltp,
                        'close': current_ltp,
                        'time_str': ist_now.strftime('%H:%M:%S')
                    }
                else:
                    # Use previous close as current open
                    prev_close = st.session_state[data_key][-1]['close']
                    
                    # Simulate intraday high/low based on LTP movement
                    price_change = abs(current_ltp - prev_close)
                    volatility_factor = max(0.01, price_change / prev_close) if prev_close > 0 else 0.01
                    
                    high = current_ltp * (1 + volatility_factor * 0.5)
                    low = current_ltp * (1 - volatility_factor * 0.5)
                    
                    ohlc_point = {
                        'timestamp': ist_now,
                        'open': prev_close,
                        'high': max(high, prev_close, current_ltp),
                        'low': min(low, prev_close, current_ltp),
                        'close': current_ltp,
                        'time_str': ist_now.strftime('%H:%M:%S')
                    }
                
                # Add new data point
                st.session_state[data_key].append(ohlc_point)
                
                # Keep only last max_points
                if len(st.session_state[data_key]) > max_points:
                    st.session_state[data_key] = st.session_state[data_key][-max_points:]
                
                return st.session_state[data_key], current_ltp
    
    return [], 0

def create_candlestick_chart(ohlc_data, strike_price, option_type, st_period, st_multiplier):
    """Create candlestick chart with SuperTrend from real data"""
    
    if len(ohlc_data) < 4:
        fig = go.Figure()
        fig.add_annotation(
            text="Collecting data... Please wait for more data points",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="white")
        )
        return fig, None
    
    df = pd.DataFrame(ohlc_data)
    
    # Ensure all OHLC values are numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill')
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=list(range(len(df))),
        open=df['open'].tolist(),
        high=df['high'].tolist(),
        low=df['low'].tolist(),
        close=df['close'].tolist(),
        name=f"{strike_price} {option_type}",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='rgba(0,255,136,0.3)',
        decreasing_fillcolor='rgba(255,68,68,0.3)'
    ))
    
    # Calculate SuperTrend if enough data
    if len(df) >= st_period + 2:
        supertrend, trend_direction = calculate_supertrend(
            df['high'].tolist(),
            df['low'].tolist(), 
            df['close'].tolist(),
            period=st_period,
            multiplier=st_multiplier
        )
        
        if supertrend and len(supertrend) > 0:
            # Prepare SuperTrend data
            st_up = []
            st_down = []
            trend_change_points = []
            
            for i, (st_val, trend) in enumerate(zip(supertrend, trend_direction)):
                if trend == 1:  # Bullish
                    st_up.append(st_val)
                    st_down.append(None)
                    
                    # Check for trend change
                    if i > 0 and trend_direction[i-1] == -1:
                        trend_change_points.append({'index': i, 'value': st_val, 'type': 'bullish'})
                else:  # Bearish
                    st_up.append(None)
                    st_down.append(st_val)
                    
                    # Check for trend change
                    if i > 0 and trend_direction[i-1] == 1:
                        trend_change_points.append({'index': i, 'value': st_val, 'type': 'bearish'})
            
            # Add SuperTrend lines
            x_values = list(range(len(supertrend)))
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=st_up,
                mode='lines',
                name='SuperTrend Up',
                line=dict(color='#00ff88', width=3),
                connectgaps=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=st_down,
                mode='lines',
                name='SuperTrend Down',
                line=dict(color='#ff4444', width=3),
                connectgaps=False
            ))
            
            # Add trend change arrows
            for point in trend_change_points:
                if point['type'] == 'bullish':
                    fig.add_annotation(
                        x=point['index'],
                        y=point['value'],
                        text="▲",
                        showarrow=False,
                        font=dict(color='#00ff88', size=16),
                        bgcolor='rgba(0,255,136,0.2)',
                        bordercolor='#00ff88',
                        borderwidth=1
                    )
                else:
                    fig.add_annotation(
                        x=point['index'],
                        y=point['value'],
                        text="▼",
                        showarrow=False,
                        font=dict(color='#ff4444', size=16),
                        bgcolor='rgba(255,68,68,0.2)',
                        bordercolor='#ff4444',
                        borderwidth=1
                    )
            
            # Check for recent trend change
            if len(trend_direction) >= 2:
                current_trend = trend_direction[-1]
                previous_trend = trend_direction[-2]
                
                if current_trend != previous_trend:
                    return fig, current_trend
    
    # Update layout
    fig.update_layout(
        title=f"{strike_price} {option_type} - Real-time LTP with SuperTrend<br><sub style='color:#888'>Live data points: {len(df)}</sub>",
        template='plotly_dark',
        height=600,
        xaxis_title="Data Points",
        yaxis_title="Price (₹)",
        showlegend=True,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, len(df), max(1, len(df)//10))),
            ticktext=[df['time_str'].iloc[i] for i in range(0, len(df), max(1, len(df)//10))]
        )
    )
    
    return fig, None

def main():
    st.title("Nifty Options SuperTrend Trader - Real LTP Data")
    
    # Initialize session state
    if 'alert_sent' not in st.session_state:
        st.session_state.alert_sent = set()
    
    api = DhanAPI()
    telegram = TelegramBot()
    
    # Sidebar settings
    st.sidebar.header("SuperTrend Settings")
    st_period = st.sidebar.slider("Period", 5, 25, 10)
    st_multiplier = st.sidebar.slider("Multiplier", 1.0, 5.0, 3.0, 0.1)
    
    st.sidebar.header("Strike Selection")
    selected_strike = st.sidebar.selectbox(
        "Select Strike",
        ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2'],
        index=2
    )
    option_type = st.sidebar.selectbox("Option Type", ['CE', 'PE'])
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [5, 10, 15, 30], index=1)
    
    # Get expiry dates
    expiry_data = api.get_expiry_list(13, "IDX_I")
    if expiry_data and 'data' in expiry_data:
        expiry = st.sidebar.selectbox("Expiry", expiry_data['data'])
    else:
        expiry = "2024-10-31"
    
    # Control buttons
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("Refresh Now"):
            st.rerun()
    
    with col_btn2:
        if st.button("Clear Data"):
            # Clear stored data
            for key in list(st.session_state.keys()):
                if key.startswith('live_data_'):
                    del st.session_state[key]
            st.success("Data cleared!")
            st.rerun()
    
    # Get option chain data
    option_chain = api.get_option_chain(13, "IDX_I", expiry)
    
    if option_chain and 'data' in option_chain:
        spot_price = option_chain['data'].get('last_price', 25000)
        atm_strike = round(spot_price / 50) * 50
        
        # Calculate actual strike price
        if selected_strike == 'ATM':
            strike_price = atm_strike
        elif selected_strike.startswith('ATM+'):
            offset = int(selected_strike.replace('ATM+', ''))
            strike_price = atm_strike + (offset * 50)
        elif selected_strike.startswith('ATM-'):
            offset = int(selected_strike.replace('ATM-', ''))
            strike_price = atm_strike - (offset * 50)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nifty Spot", f"₹{spot_price:.2f}")
        with col2:
            st.metric("ATM Strike", atm_strike)
        with col3:
            ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
            st.metric("IST Time", ist_time)
        with col4:
            st.metric("Selected Strike", f"{strike_price} {option_type}")
        
        # Collect live data and create chart
        live_data, current_ltp = collect_live_data(api, strike_price, option_type, expiry)
        
        if current_ltp > 0:
            # Calculate moneyness
            if option_type == 'CE':
                moneyness = spot_price - strike_price
                status = f"ITM by ₹{moneyness:.2f}" if moneyness > 0 else f"OTM by ₹{abs(moneyness):.2f}"
                status_color = "🟢" if moneyness > 0 else "🔴"
            else:  # PE
                moneyness = strike_price - spot_price
                status = f"ITM by ₹{moneyness:.2f}" if moneyness > 0 else f"OTM by ₹{abs(moneyness):.2f}"
                status_color = "🟢" if moneyness > 0 else "🔴"
            
            st.subheader(f"Real LTP: ₹{current_ltp:.2f} {status_color} {status}")
            st.caption(f"Data Points Collected: {len(live_data)}")
            
            # Create chart
            chart_key = f"{strike_price}_{option_type}_{st_period}_{st_multiplier}"
            
            fig, trend_change = create_candlestick_chart(
                live_data, strike_price, option_type, st_period, st_multiplier
            )
            
            # Check for trend change and send alert
            if trend_change is not None:
                alert_key = f"{chart_key}_{trend_change}_{len(live_data)}"
                
                if alert_key not in st.session_state.alert_sent:
                    trend_text = "BULLISH ⬆️" if trend_change == 1 else "BEARISH ⬇️"
                    
                    message = f"""
<b>SuperTrend Alert!</b>

<b>Strike:</b> {strike_price} {option_type}
<b>Signal:</b> {trend_text}
<b>Real LTP:</b> ₹{current_ltp:.2f}
<b>Time:</b> {ist_time} IST
<b>Nifty:</b> ₹{spot_price:.2f}
<b>Status:</b> {status}

<i>Settings: Period={st_period}, Multiplier={st_multiplier}</i>
                    """.strip()
                    
                    if telegram.send_message(message):
                        st.success(f"Alert sent: {trend_text}")
                        st.session_state.alert_sent.add(alert_key)
                    else:
                        st.warning("Telegram notification failed")
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            # Display real-time info
            if len(live_data) > 1:
                last_change = live_data[-1]['close'] - live_data[-2]['close']
                change_pct = (last_change / live_data[-2]['close']) * 100
                change_color = "🟢" if last_change > 0 else "🔴"
                
                st.info(f"Last Change: {change_color} ₹{last_change:.2f} ({change_pct:+.2f}%) | SuperTrend: Period={st_period}, Multiplier={st_multiplier}")
            
        else:
            st.error(f"No real-time data available for {strike_price} {option_type}")
    
    else:
        st.error("Failed to fetch option chain data")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
