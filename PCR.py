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
    
    def get_historical_data(self, security_id, exchange_segment, instrument, from_date, to_date, interval="5"):
        url = f"{self.base_url}/charts/intraday"
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "fromDate": from_date,
            "toDate": to_date
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
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

def get_historical_option_data(api, strike_price, option_type, expiry):
    """Get 1 day historical data for option"""
    ist = pytz.timezone('Asia/Kolkata')
    end_time = datetime.now(ist)
    start_time = end_time - timedelta(days=1)
    
    from_date = start_time.strftime('%Y-%m-%d 09:15:00')
    to_date = end_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # For demo, we'll create synthetic historical data based on current price
    # In real implementation, you'd need option security IDs for historical API
    
    timestamps = []
    prices = []
    
    # Generate 1 day of 5-minute interval data
    current_time = start_time.replace(hour=9, minute=15)
    base_price = 50.0  # Base price for simulation
    
    while current_time <= end_time:
        if 9 <= current_time.hour < 16:  # Market hours
            # Simulate price movement
            noise = np.random.normal(0, 2)
            base_price = max(0.05, base_price + noise)
            
            timestamps.append(current_time)
            prices.append(base_price)
        
        current_time += timedelta(minutes=5)
    
    return timestamps, prices

def create_option_candlestick_chart(timestamps, prices, strike_price, option_type, st_period, st_multiplier):
    """Create candlestick chart with SuperTrend"""
    if len(prices) < 4:
        return go.Figure()
    
    # Create OHLC data from prices (5-minute intervals)
    ohlc_data = []
    for i in range(0, len(prices), 1):  # Each point becomes a candle
        if i == 0:
            open_price = prices[i]
        else:
            open_price = prices[i-1]
        
        high = prices[i] * (1 + np.random.uniform(0, 0.02))
        low = prices[i] * (1 - np.random.uniform(0, 0.02))
        close = prices[i]
        
        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        ohlc_data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    df = pd.DataFrame(ohlc_data)
    df['time_str'] = df['timestamp'].dt.strftime('%H:%M')
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['time_str'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=f"{strike_price} {option_type}",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff8840',
        decreasing_fillcolor='#ff444440'
    ))
    
    # Calculate and add SuperTrend
    if len(df) >= st_period:
        supertrend, trend_direction = calculate_supertrend(
            df['high'].tolist(), 
            df['low'].tolist(), 
            df['close'].tolist(),
            period=st_period,
            multiplier=st_multiplier
        )
        
        if supertrend:
            # Prepare SuperTrend data
            st_up = []
            st_down = []
            arrows_up = []
            arrows_down = []
            
            for i, (st_val, trend) in enumerate(zip(supertrend, trend_direction)):
                if trend == 1:  # Bullish
                    st_up.append(st_val)
                    st_down.append(None)
                    
                    # Check for trend change (arrow)
                    if i > 0 and trend_direction[i-1] == -1:
                        arrows_up.append({'x': df['time_str'].iloc[i], 'y': st_val})
                    else:
                        arrows_up.append(None)
                    arrows_down.append(None)
                else:  # Bearish
                    st_up.append(None)
                    st_down.append(st_val)
                    
                    # Check for trend change (arrow)
                    if i > 0 and trend_direction[i-1] == 1:
                        arrows_down.append({'x': df['time_str'].iloc[i], 'y': st_val})
                    else:
                        arrows_down.append(None)
                    arrows_up.append(None)
            
            # Add SuperTrend lines
            fig.add_trace(go.Scatter(
                x=df['time_str'],
                y=st_up,
                mode='lines',
                name='SuperTrend Up',
                line=dict(color='#00ff88', width=2),
                connectgaps=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df['time_str'],
                y=st_down,
                mode='lines',
                name='SuperTrend Down',
                line=dict(color='#ff4444', width=2),
                connectgaps=False
            ))
            
            # Add arrows for trend changes
            for arrow in arrows_up:
                if arrow:
                    fig.add_annotation(
                        x=arrow['x'],
                        y=arrow['y'],
                        text="▲",
                        showarrow=False,
                        font=dict(color='#00ff88', size=20),
                        bgcolor='rgba(0,255,136,0.3)',
                        bordercolor='#00ff88'
                    )
            
            for arrow in arrows_down:
                if arrow:
                    fig.add_annotation(
                        x=arrow['x'],
                        y=arrow['y'],
                        text="▼",
                        showarrow=False,
                        font=dict(color='#ff4444', size=20),
                        bgcolor='rgba(255,68,68,0.3)',
                        bordercolor='#ff4444'
                    )
            
            # Check for recent trend change for Telegram alert
            if len(trend_direction) >= 2:
                current_trend = trend_direction[-1]
                previous_trend = trend_direction[-2]
                
                if current_trend != previous_trend:
                    return fig, current_trend, df['close'].iloc[-1]
    
    fig.update_layout(
        title=f"{strike_price} {option_type} - 1 Day Chart with SuperTrend",
        template='plotly_dark',
        height=600,
        xaxis_title="Time (IST)",
        yaxis_title="Price (₹)",
        showlegend=True,
        xaxis=dict(type='category')
    )
    
    return fig, None, None

def main():
    st.title("Nifty Options SuperTrend Trader")
    
    # Initialize session state
    if 'alert_sent' not in st.session_state:
        st.session_state.alert_sent = {}
    
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
    
    # Main content
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
                
                # Get historical data and create chart
                timestamps, historical_prices = get_historical_option_data(
                    api, strike_price, option_type, expiry
                )
                
                # Add current price to historical data
                timestamps.append(datetime.now(pytz.timezone('Asia/Kolkata')))
                historical_prices.append(current_ltp)
                
                # Create chart with unique key to avoid duplicate element error
                chart_key = f"{strike_price}_{option_type}_{expiry}"
                
                chart_result = create_option_candlestick_chart(
                    timestamps, historical_prices, strike_price, option_type, 
                    st_period, st_multiplier
                )
                
                if isinstance(chart_result, tuple) and len(chart_result) == 3:
                    fig, trend_change, current_price = chart_result
                    
                    # Check for SuperTrend signal and send Telegram alert
                    if trend_change is not None:
                        alert_key = f"{chart_key}_{trend_change}_{len(timestamps)}"
                        
                        if alert_key not in st.session_state.alert_sent:
                            trend_text = "BULLISH ⬆️" if trend_change == 1 else "BEARISH ⬇️"
                            
                            message = f"""
<b>🔔 SuperTrend Alert!</b>

<b>Strike:</b> {strike_price} {option_type}
<b>Signal:</b> {trend_text}
<b>Price:</b> ₹{current_price:.2f}
<b>Time:</b> {ist_time} IST
<b>Nifty:</b> ₹{spot_price:.2f}

<i>Settings: Period={st_period}, Multiplier={st_multiplier}</i>
                            """.strip()
                            
                            if telegram.send_message(message):
                                st.success(f"📱 Alert sent: {trend_text}")
                                st.session_state.alert_sent[alert_key] = True
                            else:
                                st.warning("⚠️ Telegram notification failed")
                else:
                    fig = chart_result[0] if isinstance(chart_result, tuple) else chart_result
                
                # Display chart with unique key
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
                # Display SuperTrend settings
                st.info(f"SuperTrend Settings: Period = {st_period}, Multiplier = {st_multiplier}")
                
            else:
                st.error(f"No price data available for {strike_price} {option_type}")
        else:
            st.error(f"Strike {strike_price} not found in option chain")
    
    else:
        st.error("Failed to fetch option chain data")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
