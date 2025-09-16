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
                days_to_expiry = max(1, (end_date - datetime.fromtimestamp(timestamp/1000, ist)).days)
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
                
                dt = datetime.fromtimestamp(timestamp/1000, ist)
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

def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate SuperTrend indicator"""
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
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None
    
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
        expiry = "2025-10-31"
    
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
                        'oi': stake_data.get('pe', {}).get('oi', 0)
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
            
            # Create dropdown for selecting which strike to display
            option_choices = [f"{opt['strike']} {opt['type']}" for opt in active_options]
            selected_option = st.selectbox("Select Strike to Display", option_choices, 
                                         index=0, key="strike_selector")
            
            # Update session state with selected option
            st.session_state.selected_option = selected_option
            
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
            
            # Create chart for selected option
            if st.session_state.selected_option:
                # Parse selected option
                parts = st.session_state.selected_option.split()
                strike = int(parts[0])
                opt_type = parts[1]
                option_key = f"{strike}_{opt_type}"
                
                if option_key in st.session_state.options_data:
                    data = st.session_state.options_data[option_key]
                    
                    if len(data) > 0:
                        df = pd.DataFrame(data)
                        
                        # Create figure with candlestick chart
                        fig = go.Figure()
                        
                        # Add candlestick
                        fig.add_trace(go.Candlestick(
                            x=df['timestamp'],
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            name=f"{strike} {opt_type}",
                            increasing_line_color='#2E7D32',  # Green for up
                            decreasing_line_color='#D32F2F',  # Red for down
                        ))
                        
                        # Calculate and add SuperTrend
                        if len(df) >= st_period + 2:
                            supertrend, trend_direction = calculate_supertrend(
                                df['high'].tolist(), df['low'].tolist(), df['close'].tolist(),
                                period=st_period, multiplier=st_multiplier
                            )
                            
                            if supertrend:
                                # Add SuperTrend line with up/down arrows
                                buy_signals = []
                                sell_signals = []
                                
                                for i in range(1, len(trend_direction)):
                                    if trend_direction[i] == 1 and trend_direction[i-1] == -1:
                                        buy_signals.append((df['timestamp'].iloc[i], supertrend[i]))
                                    elif trend_direction[i] == -1 and trend_direction[i-1] == 1:
                                        sell_signals.append((df['timestamp'].iloc[i], supertrend[i]))
                                
                                # Add SuperTrend line
                                fig.add_trace(go.Scatter(
                                    x=df['timestamp'],
                                    y=supertrend,
                                    mode='lines',
                                    name=f'SuperTrend {st_period} {st_multiplier}',
                                    line=dict(color='#7B1FA2', width=2),
                                    hoverinfo='skip'
                                ))
                                
                                # Add buy signals (up arrows)
                                if buy_signals:
                                    buy_x, buy_y = zip(*buy_signals)
                                    fig.add_trace(go.Scatter(
                                        x=buy_x,
                                        y=buy_y,
                                        mode='markers',
                                        name='Buy Signal',
                                        marker=dict(
                                            symbol='triangle-up',
                                            color='#00C853',
                                            size=12,
                                            line=dict(width=2, color='#00C853')
                                        ),
                                        hovertemplate='Buy Signal<extra></extra>'
                                    ))
                                
                                # Add sell signals (down arrows)
                                if sell_signals:
                                    sell_x, sell_y = zip(*sell_signals)
                                    fig.add_trace(go.Scatter(
                                        x=sell_x,
                                        y=sell_y,
                                        mode='markers',
                                        name='Sell Signal',
                                        marker=dict(
                                            symbol='triangle-down',
                                            color='#FF5252',
                                            size=12,
                                            line=dict(width=2, color='#FF5252')
                                        ),
                                        hovertemplate='Sell Signal<extra></extra>'
                                    ))
                                
                                # Add current SuperTrend value annotation
                                current_st = supertrend[-1]
                                fig.add_annotation(
                                    x=df['timestamp'].iloc[-1],
                                    y=current_st,
                                    text=f"SuperTrend: {current_st:.2f}",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=2,
                                    arrowcolor="#7B1FA2",
                                    ax=0,
                                    ay=-40,
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="#7B1FA2",
                                    borderwidth=1,
                                    borderpad=4
                                )
                                
                                # Check trend change
                                if len(trend_direction) >= 2 and trend_direction[-1] != trend_direction[-2]:
                                    trend_text = "BULLISH" if trend_direction[-1] == 1 else "BEARISH"
                                    alert_key = f"{option_key}_{trend_direction[-1]}_{ist_time[:5]}"
                                    
                                    if alert_key not in st.session_state.alert_sent:
                                        message = f"""
<b>SuperTrend Alert!</b>
Option: {strike} {opt_type}
Signal: {trend_text}
LTP: ₹{df['close'].iloc[-1]:.2f}
Time: {ist_time} IST
                                        """.strip()
                                        
                                        if telegram.send_message(message):
                                            st.success(f"Alert: {strike} {opt_type} - {trend_text}")
                                            st.session_state.alert_sent.add(alert_key)
                        
                        # Update layout for TBT-like appearance
                        fig.update_layout(
                            template='plotly_white',
                            height=600,
                            title=f"NIFTY {expiry.split('-')[0]} {strike} {opt_type} · SuperTrend {st_period} {st_multiplier}",
                            xaxis_title="",
                            yaxis_title="Price (₹)",
                            xaxis_rangeslider_visible=False,
                            showlegend=True,
                            font=dict(size=12),
                            margin=dict(t=80, b=50, l=50, r=50),
                            hovermode='x unified'
                        )
                        
                        # Format x-axis to show proper dates
                        fig.update_xaxes(
                            tickformat="%H:%M",
                            tickangle=0,
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray'
                        )
                        
                        fig.update_yaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray'
                        )
                        
                        # Display chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current values
                        current_ltp = df['close'].iloc[-1] if len(df) > 0 else 0
                        prev_close = df['close'].iloc[-2] if len(df) > 1 else current_ltp
                        change = current_ltp - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"Current LTP", f"₹{current_ltp:.2f}", 
                                     f"{change:+.2f} ({change_pct:+.2f}%)")
                        with col2:
                            if len(df) > 0:
                                st.metric("High", f"₹{df['high'].max():.2f}")
                        with col3:
                            if len(df) > 0:
                                st.metric("Low", f"₹{df['low'].min():.2f}")
                        
                        # Show data summary
                        total_points = len(data)
                        first_time = data[0]['timestamp'].strftime('%d-%m %H:%M') if data else "N/A"
                        last_time = data[-1]['timestamp'].strftime('%d-%m %H:%M') if data else "N/A"
                        
                        st.info(f"📊 Showing data for {strike} {opt_type}: {first_time} to {last_time} | Total points: {total_points}")
                else:
                    st.warning(f"No data available for {strike} {opt_type}")
            
            else:
                st.warning("No option selected for display")
        
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