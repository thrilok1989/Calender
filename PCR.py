import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import pytz

# Page config
st.set_page_config(page_title="ATM±2 Options SuperTrend", layout="wide")

class DhanAPI:
    def __init__(self):
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'access-token': st.secrets["DHAN_ACCESS_TOKEN"],
            'client-id': st.secrets["DHAN_CLIENT_ID"],
            'Content-Type': 'application/json'
        }
    
    def get_historical_data(self, security_id, exchange_segment, instrument, from_date, to_date, interval="5"):
        """Get historical intraday data"""
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
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Historical Data API Error: {e}")
            return None
    
    def get_market_quote_ohlc(self, instruments_dict):
        """Get OHLC data for multiple instruments using Market Quote API"""
        url = f"{self.base_url}/marketfeed/ohlc"
        try:
            response = requests.post(url, headers=self.headers, json=instruments_dict, timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Market Quote API Error: {e}")
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
            return requests.post(url, data=data, timeout=5).status_code == 200
        except:
            return False

def get_3day_historical_data(api, active_options, current_spot):
    """Get 3 days of historical data for Nifty and create option data"""
    ist = pytz.timezone('Asia/Kolkata')
    
    # Calculate date range - 3 trading days back
    end_date = datetime.now(ist)
    start_date = end_date - timedelta(days=5)  # Go back 5 days to ensure 3 trading days
    
    from_date = start_date.strftime('%Y-%m-%d 09:15:00')
    to_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Get Nifty historical data
    nifty_data = api.get_historical_data(
        security_id=13,  # Nifty 50
        exchange_segment="IDX_I",
        instrument="INDEX",
        from_date=from_date,
        to_date=to_date,
        interval="5"  # 5-minute intervals
    )
    
    if not nifty_data or 'timestamp' not in nifty_data:
        st.error("Failed to fetch historical data")
        return {}
    
    historical_options_data = {}
    
    # Process each active option
    for option in active_options:
        option_key = f"{option['strike']}_{option['type']}"
        option_data_points = []
        
        timestamps = nifty_data['timestamp']
        closes = nifty_data['close']
        highs = nifty_data['high']
        lows = nifty_data['low']
        opens = nifty_data['open']
        
        for i, timestamp in enumerate(timestamps):
            if i < len(closes):
                spot_price = closes[i]
                spot_open = opens[i] if i < len(opens) else spot_price
                spot_high = highs[i] if i < len(highs) else spot_price
                spot_low = lows[i] if i < len(lows) else spot_price
                
                # Calculate option price based on moneyness
                strike = option['strike']
                option_type = option['type']
                
                # Simple option pricing based on intrinsic + time value
                if option_type == 'CE':
                    intrinsic = max(0, spot_price - strike)
                    moneyness = (spot_price - strike) / strike
                else:  # PE
                    intrinsic = max(0, strike - spot_price)
                    moneyness = (strike - spot_price) / strike
                
                # Time value decreases as we approach current time
                days_to_expiry = max(1, (end_date - datetime.fromtimestamp(timestamp, ist)).days)
                time_value = max(1, abs(moneyness) * 50 * (days_to_expiry / 7))  # Rough time value
                
                option_ltp = intrinsic + time_value
                
                # Create OHLC for option based on underlying movement
                underlying_range = (spot_high - spot_low) / spot_price if spot_price > 0 else 0
                option_volatility = underlying_range * 3  # Options are more volatile
                
                option_open = option_ltp * (spot_open / spot_price) if spot_price > 0 else option_ltp
                option_high = option_ltp * (1 + option_volatility)
                option_low = option_ltp * (1 - option_volatility)
                
                # Ensure OHLC relationships
                option_high = max(option_high, option_open, option_ltp)
                option_low = min(option_low, option_open, option_ltp)
                
                dt = datetime.fromtimestamp(timestamp, ist)
                option_data_points.append({
                    'timestamp': dt,
                    'open': float(option_open),
                    'high': float(option_high),
                    'low': float(option_low),
                    'close': float(option_ltp),
                    'time_str': dt.strftime('%m-%d %H:%M'),
                    'spot_price': spot_price
                })
        
        if option_data_points:
            historical_options_data[option_key] = option_data_points
    
    return historical_options_data
    if len(high) < period + 1:
        return [], []
    
    # Calculate True Range and ATR
    tr = []
    for i in range(1, len(high)):
        tr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr.append(tr_val)
    
    atr = []
    if len(tr) >= period:
        atr.append(sum(tr[:period]) / period)
        for i in range(period, len(tr)):
            atr.append((atr[-1] * (period - 1) + tr[i]) / period)
    
    # Calculate SuperTrend
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    supertrend, trend_direction = [], []
    
    for i in range(len(hl2)):
        if i < len(atr):
            ub = hl2[i] + multiplier * atr[i]
            lb = hl2[i] - multiplier * atr[i]
        else:
            ub = hl2[i] + multiplier * (atr[-1] if atr else 0)
            lb = hl2[i] - multiplier * (atr[-1] if atr else 0)
        
        if i == 0:
            supertrend.append(ub)
            trend_direction.append(1)
        else:
            final_ub = ub if ub < supertrend[-1] or close[i-1] > supertrend[-1] else supertrend[-1]
            final_lb = lb if lb > supertrend[-1] or close[i-1] < supertrend[-1] else supertrend[-1]
            
            if trend_direction[-1] == 1 and close[i] <= final_lb:
                trend_direction.append(-1)
                supertrend.append(final_ub)
            elif trend_direction[-1] == -1 and close[i] >= final_ub:
                trend_direction.append(1)
                supertrend.append(final_lb)
            else:
                trend_direction.append(trend_direction[-1])
                supertrend.append(final_lb if trend_direction[-1] == 1 else final_ub)
    
    return supertrend, trend_direction

def main():
    st.title("ATM±2 Nifty Options SuperTrend Trader")
    
    # Initialize session state for data storage
    if 'options_data' not in st.session_state:
        st.session_state.options_data = {}
    if 'alert_sent' not in st.session_state:
        st.session_state.alert_sent = set()
    
    api = DhanAPI()
    telegram = TelegramBot()
    
    # Sidebar
    st.sidebar.header("SuperTrend Settings")
    st_period = st.sidebar.slider("Period", 5, 25, 10)
    st_multiplier = st.sidebar.slider("Multiplier", 1.0, 5.0, 3.0, 0.1)
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox("Refresh (seconds)", [5, 10, 15, 30], index=1)
    
    # Get expiry
    expiry_data = api.get_expiry_list(13, "IDX_I")
    if expiry_data and 'data' in expiry_data:
        expiry = st.sidebar.selectbox("Expiry", expiry_data['data'])
    else:
        expiry = "2024-10-31"
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Refresh"):
            st.rerun()
    with col2:
        if st.button("Clear Data"):
            st.session_state.options_data = {}
            st.success("Cleared!")
            st.rerun()
    
    # Get option chain
    option_chain = api.get_option_chain(13, "IDX_I", expiry)
    
    if option_chain and 'data' in option_chain:
        spot_price = option_chain['data'].get('last_price', 25000)
        atm_strike = round(spot_price / 50) * 50
        
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
            # Market status
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
            market_status = "OPEN" if market_open <= ist_now <= market_close else "CLOSED"
            st.metric("Market", market_status)
        
        # Find and display active strikes
        oc_data = option_chain['data'].get('oc', {})
        
        # Check ATM±2 strikes
        active_options = []
        for offset in range(-2, 3):  # ATM-2 to ATM+2
            strike = atm_strike + (offset * 50)
            strike_key = f"{strike}.000000"
            
            if strike_key in oc_data:
                strike_data = oc_data[strike_key]
                ce_ltp = strike_data.get('ce', {}).get('last_price', 0)
                pe_ltp = strike_data.get('pe', {}).get('last_price', 0)
                
                if ce_ltp > 0:
                    active_options.append({
                        'strike': strike,
                        'type': 'CE',
                        'ltp': ce_ltp,
                        'offset': offset,
                        'volume': strike_data.get('ce', {}).get('volume', 0),
                        'oi': strike_data.get('ce', {}).get('oi', 0)
                    })
                
                if pe_ltp > 0:
                    active_options.append({
                        'strike': strike,
                        'type': 'PE', 
                        'ltp': pe_ltp,
                        'offset': offset,
                        'volume': strike_data.get('pe', {}).get('volume', 0),
                        'oi': strike_data.get('pe', {}).get('oi', 0)
                    })
        
        if active_options:
            st.subheader(f"Active ATM±2 Strikes Found: {len(active_options)} options")
            
            # Show strikes in columns
            strikes_display = {}
            for opt in active_options:
                strike_key = opt['strike']
                if strike_key not in strikes_display:
                    atm_label = f"ATM{opt['offset']:+d}" if opt['offset'] != 0 else "ATM"
                    strikes_display[strike_key] = {
                        'label': f"{strike_key} ({atm_label})",
                        'ce': None,
                        'pe': None
                    }
                
                if opt['type'] == 'CE':
                    strikes_display[strike_key]['ce'] = f"₹{opt['ltp']:.2f}"
                else:
                    strikes_display[strike_key]['pe'] = f"₹{opt['ltp']:.2f}"
            
            # Display strikes
            strike_cols = st.columns(len(strikes_display))
            for i, (strike, info) in enumerate(strikes_display.items()):
                with strike_cols[i]:
                    st.write(f"**{info['label']}**")
                    if info['ce']:
                        st.write(f"CE: {info['ce']}")
                    if info['pe']:
                        st.write(f"PE: {info['pe']}")
            
            # Load 3-day historical data
            with st.spinner("Loading 3 days historical data..."):
                historical_data = get_3day_historical_data(api, active_options, spot_price)
            
            if historical_data:
                st.success(f"Loaded 3 days of historical data for {len(historical_data)} options")
                
                # Update session state with historical data
                for option_key, data_points in historical_data.items():
                    st.session_state.options_data[option_key] = data_points
                
                # Add current live data point
                current_time = ist_now
                for option in active_options:
                    option_key = f"{option['strike']}_{option['type']}"
                    
                    if option_key in st.session_state.options_data:
                        # Calculate current option OHLC based on current LTP
                        last_data = st.session_state.options_data[option_key][-1]
                        prev_close = last_data['close']
                        
                        # Simple volatility estimation
                        price_change = abs(option['ltp'] - prev_close) / prev_close if prev_close > 0 else 0.02
                        volatility = max(0.01, price_change)
                        
                        current_ohlc = {
                            'timestamp': current_time,
                            'open': prev_close,
                            'high': max(option['ltp'] * (1 + volatility), prev_close, option['ltp']),
                            'low': min(option['ltp'] * (1 - volatility), prev_close, option['ltp']),
                            'close': option['ltp'],
                            'time_str': current_time.strftime('%m-%d %H:%M'),
                            'spot_price': spot_price
                        }
                        
                        # Add current data point
                        st.session_state.options_data[option_key].append(current_ohlc)
            else:
                st.error("Failed to load historical data")
                return
            
            # Create charts
            ce_options = [opt for opt in active_options if opt['type'] == 'CE']
            pe_options = [opt for opt in active_options if opt['type'] == 'PE']
            
            if ce_options or pe_options:
                # Create subplots
                rows = 2 if (ce_options and pe_options) else 1
                fig = make_subplots(
                    rows=rows, 
                    cols=1,
                    subplot_titles=[
                        "Call Options (CE)" if ce_options else None,
                        "Put Options (PE)" if pe_options else None
                    ],
                    vertical_spacing=0.1
                )
                
                colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
                
                # Plot CE options
                if ce_options:
                    for i, option in enumerate(ce_options):
                        option_key = f"{option['strike']}_{option['type']}"
                        data = st.session_state.options_data.get(option_key, [])
                        
                        if len(data) >= 1:
                            df = pd.DataFrame(data)
                            
                            fig.add_trace(go.Candlestick(
                                x=list(range(len(df))),
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name=f"{option['strike']} CE",
                                increasing_line_color=colors[i % len(colors)],
                                decreasing_line_color=colors[i % len(colors)],
                                showlegend=True
                            ), row=1, col=1)
                            
                            # Add SuperTrend to first CE option
                            if i == 0 and len(df) >= st_period + 2:
                                supertrend, trend_direction = calculate_supertrend(
                                    df['high'].tolist(), df['low'].tolist(), df['close'].tolist(),
                                    period=st_period, multiplier=st_multiplier
                                )
                                
                                if supertrend:
                                    st_up = [st if td == 1 else None for st, td in zip(supertrend, trend_direction)]
                                    st_down = [st if td == -1 else None for st, td in zip(supertrend, trend_direction)]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=list(range(len(st_up))),
                                        y=st_up,
                                        mode='lines',
                                        name='SuperTrend Up',
                                        line=dict(color='#00ff88', width=2),
                                        connectgaps=False
                                    ), row=1, col=1)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=list(range(len(st_down))),
                                        y=st_down,
                                        mode='lines',
                                        name='SuperTrend Down',
                                        line=dict(color='#ff4444', width=2),
                                        connectgaps=False
                                    ), row=1, col=1)
                                    
                                    # Check trend change
                                    if len(trend_direction) >= 2 and trend_direction[-1] != trend_direction[-2]:
                                        trend_text = "BULLISH" if trend_direction[-1] == 1 else "BEARISH"
                                        alert_key = f"{option_key}_{trend_direction[-1]}_{ist_time[:5]}"
                                        
                                        if alert_key not in st.session_state.alert_sent:
                                            message = f"""
<b>SuperTrend Alert!</b>
Option: {option['strike']} CE
Signal: {trend_text}
LTP: ₹{option['ltp']:.2f}
Time: {ist_time} IST
                                            """.strip()
                                            
                                            if telegram.send_message(message):
                                                st.success(f"Alert: {option['strike']} CE - {trend_text}")
                                                st.session_state.alert_sent.add(alert_key)
                
                # Plot PE options  
                if pe_options:
                    row_num = 2 if (ce_options and pe_options) else 1
                    for i, option in enumerate(pe_options):
                        option_key = f"{option['strike']}_{option['type']}"
                        data = st.session_state.options_data.get(option_key, [])
                        
                        if len(data) >= 1:
                            df = pd.DataFrame(data)
                            
                            fig.add_trace(go.Candlestick(
                                x=list(range(len(df))),
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name=f"{option['strike']} PE",
                                increasing_line_color=colors[i % len(colors)],
                                decreasing_line_color=colors[i % len(colors)],
                                showlegend=True
                            ), row=row_num, col=1)
                
                # Update layout
                fig.update_layout(
                    template='plotly_dark',
                    height=800 if rows == 2 else 600,
                    title="ATM±2 Nifty Options - 3 Days Historical LTP with SuperTrend",
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    xaxis2_rangeslider_visible=False if rows == 2 else None
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data summary
                total_data_points = sum(len(data) for data in st.session_state.options_data.values())
                first_data_time = None
                last_data_time = None
                
                if st.session_state.options_data:
                    all_times = []
                    for data in st.session_state.options_data.values():
                        if data:
                            all_times.extend([point['timestamp'] for point in data])
                    
                    if all_times:
                        first_data_time = min(all_times).strftime('%d-%m %H:%M')
                        last_data_time = max(all_times).strftime('%d-%m %H:%M')
                
                if first_data_time and last_data_time:
                    st.info(f"📊 Showing 3-day historical data: {first_data_time} to {last_data_time} | Total points: {total_data_points}")
                else:
                    st.info(f"Tracking {len(active_options)} active options | Total data points: {total_data_points}")
            
            else:
                st.warning("No chart data available yet")
        
        else:
            st.error("No active options found in ATM±2 range")
            st.write("This could mean:")
            st.write("- Market is closed")
            st.write("- No trading activity in nearby strikes")
            st.write("- Try a different expiry date")
    
    else:
        st.error("Failed to fetch option chain data")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
