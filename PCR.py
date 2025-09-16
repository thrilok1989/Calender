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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
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

def find_active_strikes_around_atm(oc_data, atm_strike, max_range=2):
    """Find active strikes within ATM±2 range"""
    active_strikes = {}
    
    for offset in range(-max_range, max_range + 1):
        strike = atm_strike + (offset * 50)
        strike_key = f"{strike}.000000"
        
        if strike_key in oc_data:
            strike_data = oc_data[strike_key]
            ce_ltp = strike_data.get('ce', {}).get('last_price', 0)
            pe_ltp = strike_data.get('pe', {}).get('last_price', 0)
            
            if ce_ltp > 0 or pe_ltp > 0:  # At least one option type is active
                active_strikes[offset] = {
                    'strike': strike,
                    'ce_ltp': ce_ltp,
                    'pe_ltp': pe_ltp,
                    'ce_active': ce_ltp > 0,
                    'pe_active': pe_ltp > 0
                }
    
    return active_strikes

def collect_live_data_multiple(api, active_strikes, expiry, max_points=50):
    """Collect data for multiple strikes"""
    if 'multi_data' not in st.session_state:
        st.session_state.multi_data = {}
    
    option_chain = api.get_option_chain(13, "IDX_I", expiry)
    if not option_chain or 'data' not in option_chain:
        return {}
    
    oc_data = option_chain['data'].get('oc', {})
    ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
    
    current_data = {}
    
    for offset, strike_info in active_strikes.items():
        strike = strike_info['strike']
        strike_key = f"{strike}.000000"
        
        if strike_key in oc_data:
            strike_data = oc_data[strike_key]
            
            for option_type in ['ce', 'pe']:
                if strike_info[f'{option_type}_active']:
                    current_ltp = strike_data.get(option_type, {}).get('last_price', 0)
                    
                    if current_ltp > 0:
                        data_key = f"{strike}_{option_type.upper()}"
                        
                        if data_key not in st.session_state.multi_data:
                            st.session_state.multi_data[data_key] = []
                        
                        # Create OHLC point
                        if len(st.session_state.multi_data[data_key]) == 0:
                            ohlc_point = {
                                'timestamp': ist_now,
                                'open': current_ltp,
                                'high': current_ltp,
                                'low': current_ltp,
                                'close': current_ltp,
                                'time_str': ist_now.strftime('%H:%M:%S')
                            }
                        else:
                            prev_close = st.session_state.multi_data[data_key][-1]['close']
                            volatility = max(0.01, abs(current_ltp - prev_close) / prev_close)
                            
                            ohlc_point = {
                                'timestamp': ist_now,
                                'open': prev_close,
                                'high': max(current_ltp * (1 + volatility * 0.3), prev_close, current_ltp),
                                'low': min(current_ltp * (1 - volatility * 0.3), prev_close, current_ltp),
                                'close': current_ltp,
                                'time_str': ist_now.strftime('%H:%M:%S')
                            }
                        
                        st.session_state.multi_data[data_key].append(ohlc_point)
                        
                        # Keep only last max_points
                        if len(st.session_state.multi_data[data_key]) > max_points:
                            st.session_state.multi_data[data_key] = st.session_state.multi_data[data_key][-max_points:]
                        
                        current_data[data_key] = {
                            'ltp': current_ltp,
                            'data': st.session_state.multi_data[data_key],
                            'offset': offset,
                            'strike': strike,
                            'type': option_type.upper()
                        }
    
    return current_data

def create_multiple_charts(current_data, st_period, st_multiplier):
    """Create subplots for multiple options"""
    if not current_data:
        fig = go.Figure()
        fig.add_annotation(text="No active options found", x=0.5, y=0.5, showarrow=False)
        return fig, []
    
    # Separate CE and PE
    ce_data = {k: v for k, v in current_data.items() if v['type'] == 'CE'}
    pe_data = {k: v for k, v in current_data.items() if v['type'] == 'PE'}
    
    # Create subplots
    rows = 2 if pe_data else 1
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=["Call Options (CE)" + (" with SuperTrend" if ce_data else ""), 
                       "Put Options (PE)" + (" with SuperTrend" if pe_data else "")] if rows == 2 else ["Call Options (CE) with SuperTrend"],
        vertical_spacing=0.1
    )
    
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
    trend_changes = []
    
    # Plot CE options
    for i, (key, data_info) in enumerate(ce_data.items()):
        df = pd.DataFrame(data_info['data'])
        if len(df) >= 4:
            fig.add_trace(go.Candlestick(
                x=list(range(len(df))),
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=f"{data_info['strike']} CE",
                increasing_line_color=colors[i % len(colors)],
                decreasing_line_color=colors[i % len(colors)],
                showlegend=True
            ), row=1, col=1)
            
            # Add SuperTrend for first CE option
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
                    
                    # Check for trend change
                    if len(trend_direction) >= 2 and trend_direction[-1] != trend_direction[-2]:
                        trend_changes.append({
                            'option': key,
                            'trend': trend_direction[-1],
                            'ltp': data_info['ltp']
                        })
    
    # Plot PE options
    if pe_data:
        for i, (key, data_info) in enumerate(pe_data.items()):
            df = pd.DataFrame(data_info['data'])
            if len(df) >= 4:
                fig.add_trace(go.Candlestick(
                    x=list(range(len(df))),
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=f"{data_info['strike']} PE",
                    increasing_line_color=colors[i % len(colors)],
                    decreasing_line_color=colors[i % len(colors)],
                    showlegend=True
                ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        height=800 if rows == 2 else 600,
        title="ATM±2 Nifty Options - Real-time LTP with SuperTrend",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False if rows == 2 else None
    )
    
    return fig, trend_changes

def main():
    st.title("ATM±2 Nifty Options SuperTrend Trader")
    
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
            st.session_state.multi_data = {}
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
        
        # Find active strikes around ATM
        oc_data = option_chain['data'].get('oc', {})
        active_strikes = find_active_strikes_around_atm(oc_data, atm_strike)
        
        if active_strikes:
            st.subheader("Active ATM±2 Strikes Found:")
            
            # Show active strikes info
            strike_info_cols = st.columns(len(active_strikes))
            for i, (offset, strike_info) in enumerate(active_strikes.items()):
                with strike_info_cols[i]:
                    atm_label = f"ATM{offset:+d}" if offset != 0 else "ATM"
                    st.write(f"**{strike_info['strike']} ({atm_label})**")
                    if strike_info['ce_active']:
                        st.write(f"CE: ₹{strike_info['ce_ltp']:.2f}")
                    if strike_info['pe_active']:
                        st.write(f"PE: ₹{strike_info['pe_ltp']:.2f}")
            
            # Collect and display data
            current_data = collect_live_data_multiple(api, active_strikes, expiry)
            
            if current_data:
                fig, trend_changes = create_multiple_charts(current_data, st_period, st_multiplier)
                st.plotly_chart(fig, use_container_width=True)
                
                # Handle trend change alerts
                for change in trend_changes:
                    alert_key = f"{change['option']}_{change['trend']}_{ist_time[:5]}"
                    if alert_key not in st.session_state.alert_sent:
                        trend_text = "BULLISH" if change['trend'] == 1 else "BEARISH"
                        message = f"""
<b>SuperTrend Alert!</b>
Option: {change['option']}
Signal: {trend_text}
LTP: ₹{change['ltp']:.2f}
Time: {ist_time} IST
Nifty: ₹{spot_price:.2f}
                        """.strip()
                        
                        if telegram.send_message(message):
                            st.success(f"Alert: {change['option']} - {trend_text}")
                            st.session_state.alert_sent.add(alert_key)
                
                # Show summary
                total_points = sum(len(data['data']) for data in current_data.values())
                st.info(f"Tracking {len(current_data)} active options | Total data points: {total_points}")
            
            else:
                st.warning("Found strikes but no live data available")
        
        else:
            st.error("No active strikes found in ATM±2 range")
            st.write("This usually means:")
            st.write("• Market is closed")
            st.write("• No trading activity in nearby strikes")
            st.write("• Try different expiry date")
    
    else:
        st.error("Failed to fetch option chain data")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()