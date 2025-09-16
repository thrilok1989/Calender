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

def generate_historical_data(current_price, points=78):
    """Generate 1 day of historical OHLC data (5-min intervals)"""
    np.random.seed(42)  # For consistent data
    
    prices = []
    base_price = current_price * 0.8  # Start from 80% of current price
    
    for i in range(points):
        # Simulate price movement
        change = np.random.normal(0, base_price * 0.02)
        base_price = max(0.05, base_price + change)
        prices.append(base_price)
    
    # Generate OHLC data
    ohlc_data = []
    ist = pytz.timezone('Asia/Kolkata')
    start_time = datetime.now(ist).replace(hour=9, minute=15, second=0, microsecond=0)
    
    for i, price in enumerate(prices):
        timestamp = start_time + timedelta(minutes=i * 5)
        
        # Create OHLC with some randomness
        volatility = 0.015
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        
        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1]
        
        close_price = price
        
        # Ensure OHLC relationships
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        ohlc_data.append({
            'timestamp': timestamp,
            'open': float(open_price),
            'high': float(high),
            'low': float(low),
            'close': float(close_price),
            'time_str': timestamp.strftime('%H:%M')
        })
    
    return ohlc_data

def create_candlestick_chart(ohlc_data, strike_price, option_type, st_period, st_multiplier):
    """Create candlestick chart with SuperTrend"""
    
    if len(ohlc_data) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", x=0.5, y=0.5, showarrow=False)
        return fig, None
    
    df = pd.DataFrame(ohlc_data)
    
    # Ensure all OHLC values are numeric and valid
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(method='ffill').fillna(0)
    
    # Validate OHLC relationships
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=list(range(len(df))),  # Use numeric index instead of time strings
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
    
    # Calculate SuperTrend
    if len(df) >= st_period + 5:
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
        title=f"{strike_price} {option_type} - 1 Day Chart with SuperTrend",
        template='plotly_dark',
        height=600,
        xaxis_title="Time Points",
        yaxis_title="Price (₹)",
        showlegend=True,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, len(df), 10)),
            ticktext=[df['time_str'].iloc[i] for i in range(0, len(df), 10)]
        )
    )
    
    return fig, None

def main():
    st.title("Nifty Options SuperTrend Trader")
    
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
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [10, 30, 60], index=1)
    
    # Get expiry dates
    expiry_data = api.get_expiry_list(13, "IDX_I")
    if expiry_data and 'data' in expiry_data:
        expiry = st.sidebar.selectbox("Expiry", expiry_data['data'])
    else:
        expiry = "2024-10-31"
    
    # Manual refresh
    if st.sidebar.button("Refresh Data"):
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
        
        # Get current option price
        oc_data = option_chain['data'].get('oc', {})
        strike_key = f"{strike_price}.000000"
        
        if strike_key in oc_data:
            option_data = oc_data[strike_key].get(option_type.lower(), {})
            current_ltp = option_data.get('last_price', 0)
            
            if current_ltp > 0:
                st.subheader(f"Current LTP: ₹{current_ltp:.2f}")
                
                # Generate historical data
                historical_data = generate_historical_data(current_ltp)
                
                # Create chart
                chart_key = f"{strike_price}_{option_type}_{st_period}_{st_multiplier}"
                
                fig, trend_change = create_candlestick_chart(
                    historical_data, strike_price, option_type, st_period, st_multiplier
                )
                
                # Check for trend change and send alert
                if trend_change is not None:
                    alert_key = f"{chart_key}_{trend_change}_{ist_time[:5]}"  # Include time to avoid duplicate alerts
                    
                    if alert_key not in st.session_state.alert_sent:
                        trend_text = "BULLISH ⬆️" if trend_change == 1 else "BEARISH ⬇️"
                        
                        message = f"""
<b>SuperTrend Alert!</b>

<b>Strike:</b> {strike_price} {option_type}
<b>Signal:</b> {trend_text}
<b>Price:</b> ₹{current_ltp:.2f}
<b>Time:</b> {ist_time} IST
<b>Nifty:</b> ₹{spot_price:.2f}

<i>Settings: Period={st_period}, Multiplier={st_multiplier}</i>
                        """.strip()
                        
                        if telegram.send_message(message):
                            st.success(f"Alert sent: {trend_text}")
                            st.session_state.alert_sent.add(alert_key)
                        else:
                            st.warning("Telegram notification failed")
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
                # Settings info
                st.info(f"SuperTrend: Period={st_period}, Multiplier={st_multiplier} | Arrows show trend changes")
                
            else:
                st.error(f"No price data for {strike_price} {option_type}")
        else:
            st.error(f"Strike {strike_price} not found")
    else:
        st.error("Failed to fetch option chain data")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
