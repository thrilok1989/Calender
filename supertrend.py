import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pytz import timezone
import time
import os
import numpy as np

# Config
st.set_page_config(page_title="ATM SuperTrend Chart", layout="wide")
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", os.getenv("DHAN_CLIENT_ID", ""))
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", os.getenv("DHAN_ACCESS_TOKEN", ""))

if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
    st.error("Configure DHAN credentials in .streamlit/secrets.toml")
    st.stop()

# Session state for price history
if 'price_history' not in st.session_state:
    st.session_state.price_history = {
        'atm_call': [], 'atm_put': [], 'atm_plus_call': [], 'atm_plus_put': [],
        'atm_minus_call': [], 'atm_minus_put': [], 'timestamps': []
    }

# Auto refresh
st_autorefresh(interval=15000, key="refresh")  # 15 seconds for chart data

def dhan_api(endpoint, payload):
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    try:
        response = requests.post(f"https://api.dhan.co/v2{endpoint}", headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def is_market_open():
    now = datetime.now(timezone("Asia/Kolkata"))
    return now.weekday() < 5 and datetime.strptime("09:15", "%H:%M").time() <= now.time() <= datetime.strptime("15:30", "%H:%M").time()

def calculate_atr(high, low, close, period=10):
    """Calculate Average True Range"""
    if len(high) < period + 1:
        return [0] * len(high)
    
    tr_values = []
    for i in range(1, len(high)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr_values.append(max(tr1, tr2, tr3))
    
    # Calculate ATR using SMA
    atr_values = []
    for i in range(len(tr_values)):
        if i < period - 1:
            atr_values.append(0)
        else:
            atr_values.append(sum(tr_values[i-period+1:i+1]) / period)
    
    return [0] + atr_values  # Add 0 for first element

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate SuperTrend indicator"""
    if len(high) < period + 1:
        return [0] * len(high), [0] * len(high), [1] * len(high)
    
    # Calculate basic bands
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    atr = calculate_atr(high, low, close, period)
    
    upper_band = [hl2[i] + (multiplier * atr[i]) for i in range(len(hl2))]
    lower_band = [hl2[i] - (multiplier * atr[i]) for i in range(len(hl2))]
    
    # Calculate final bands
    final_upper = [0] * len(upper_band)
    final_lower = [0] * len(lower_band)
    supertrend = [0] * len(close)
    trend = [1] * len(close)  # 1 = uptrend, -1 = downtrend
    
    for i in range(len(close)):
        if i == 0:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
            supertrend[i] = upper_band[i]
        else:
            # Final upper band
            final_upper[i] = upper_band[i] if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1] else final_upper[i-1]
            
            # Final lower band
            final_lower[i] = lower_band[i] if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1] else final_lower[i-1]
            
            # SuperTrend
            if trend[i-1] == 1:  # Previous uptrend
                if close[i] <= final_lower[i]:
                    supertrend[i] = final_upper[i]
                    trend[i] = -1
                else:
                    supertrend[i] = final_lower[i]
                    trend[i] = 1
            else:  # Previous downtrend
                if close[i] >= final_upper[i]:
                    supertrend[i] = final_lower[i]
                    trend[i] = 1
                else:
                    supertrend[i] = final_upper[i]
                    trend[i] = -1
    
    return supertrend, trend, final_upper, final_lower

def get_option_data():
    """Get ATM and ATM±1 option data"""
    try:
        # Get NIFTY spot price first
        spot_data = dhan_api("/marketfeed/ltp", {"IDX_I": [13]})
        if not spot_data or 'data' not in spot_data:
            return None
        
        spot_price = spot_data['data']['IDX_I']['13']['last_price']
        
        # Get option chain
        expiry_data = dhan_api("/optionchain/expirylist", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"})
        if not expiry_data:
            return None
        
        oc_data = dhan_api("/optionchain", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry_data['data'][0]})
        if not oc_data:
            return None
        
        # Find ATM strike
        strikes = [float(strike) for strike in oc_data['data']['oc'].keys()]
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        atm_plus_strike = atm_strike + 50
        atm_minus_strike = atm_strike - 50
        
        # Extract option prices
        oc = oc_data['data']['oc']
        
        result = {
            'spot_price': spot_price,
            'atm_strike': atm_strike,
            'timestamp': datetime.now(),
            'atm_call': oc.get(str(atm_strike), {}).get('ce', {}).get('last_price', 0),
            'atm_put': oc.get(str(atm_strike), {}).get('pe', {}).get('last_price', 0),
            'atm_plus_call': oc.get(str(atm_plus_strike), {}).get('ce', {}).get('last_price', 0),
            'atm_plus_put': oc.get(str(atm_plus_strike), {}).get('pe', {}).get('last_price', 0),
            'atm_minus_call': oc.get(str(atm_minus_strike), {}).get('ce', {}).get('last_price', 0),
            'atm_minus_put': oc.get(str(atm_minus_strike), {}).get('pe', {}).get('last_price', 0)
        }
        
        return result
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def update_price_history(data):
    """Update price history with new data"""
    if not data:
        return
    
    # Add new data point
    st.session_state.price_history['atm_call'].append(data['atm_call'])
    st.session_state.price_history['atm_put'].append(data['atm_put'])
    st.session_state.price_history['atm_plus_call'].append(data['atm_plus_call'])
    st.session_state.price_history['atm_plus_put'].append(data['atm_plus_put'])
    st.session_state.price_history['atm_minus_call'].append(data['atm_minus_call'])
    st.session_state.price_history['atm_minus_put'].append(data['atm_minus_put'])
    st.session_state.price_history['timestamps'].append(data['timestamp'])
    
    # Keep only last 100 data points
    max_points = 100
    for key in st.session_state.price_history:
        if len(st.session_state.price_history[key]) > max_points:
            st.session_state.price_history[key] = st.session_state.price_history[key][-max_points:]

def create_chart(option_type, strike_label):
    """Create SuperTrend chart for specific option"""
    history = st.session_state.price_history
    
    if len(history['timestamps']) < 10:
        return None
    
    prices = history[option_type]
    timestamps = history['timestamps']
    
    # For SuperTrend, we need OHLC. Using LTP as close, creating synthetic OHLC
    close_prices = prices
    high_prices = prices  # Simplified - using LTP as high
    low_prices = prices   # Simplified - using LTP as low
    
    # Calculate SuperTrend
    supertrend, trend, upper_band, lower_band = calculate_supertrend(high_prices, low_prices, close_prices)
    
    # Find trend change points for arrows
    arrows_x, arrows_y, arrows_text, arrows_color = [], [], [], []
    for i in range(1, len(trend)):
        if trend[i] != trend[i-1]:  # Trend change
            arrows_x.append(timestamps[i])
            arrows_y.append(close_prices[i])
            if trend[i] == 1:  # Changed to uptrend
                arrows_text.append("▲")
                arrows_color.append("green")
            else:  # Changed to downtrend
                arrows_text.append("▼")
                arrows_color.append("red")
    
    # Create chart
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=close_prices,
        mode='lines',
        name=f'{strike_label} LTP',
        line=dict(color='blue', width=2)
    ))
    
    # Add SuperTrend
    supertrend_colors = ['green' if t == 1 else 'red' for t in trend]
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=supertrend,
        mode='lines',
        name='SuperTrend',
        line=dict(color='purple', width=2)
    ))
    
    # Add trend change arrows
    if arrows_x:
        fig.add_trace(go.Scatter(
            x=arrows_x,
            y=arrows_y,
            mode='markers+text',
            text=arrows_text,
            textposition="top center",
            marker=dict(size=15, color=arrows_color),
            name='Trend Signals',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{strike_label} with SuperTrend',
        xaxis_title='Time',
        yaxis_title='Price (₹)',
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def main():
    st.title("ATM Options SuperTrend Chart")
    
    if not is_market_open():
        st.warning("Market Closed")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("SuperTrend Settings")
        period = st.slider("Period", 5, 20, 10)
        multiplier = st.slider("Multiplier", 1.0, 5.0, 3.0, 0.1)
        
        st.subheader("Chart Info")
        st.write("• Green arrows: Trend turns bullish")
        st.write("• Red arrows: Trend turns bearish")
        st.write("• Purple line: SuperTrend")
        st.write("• Blue line: Option LTP")
        
        if st.button("Reset Data", use_container_width=True):
            st.session_state.price_history = {
                'atm_call': [], 'atm_put': [], 'atm_plus_call': [], 'atm_plus_put': [],
                'atm_minus_call': [], 'atm_minus_put': [], 'timestamps': []
            }
            st.success("Data reset!")
    
    # Get current data
    current_data = get_option_data()
    if current_data:
        update_price_history(current_data)
        
        # Display current info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("NIFTY Spot", f"{current_data['spot_price']:.2f}")
        with col2:
            st.metric("ATM Strike", current_data['atm_strike'])
        with col3:
            st.metric("Data Points", len(st.session_state.price_history['timestamps']))
        with col4:
            st.metric("Last Update", current_data['timestamp'].strftime("%H:%M:%S"))
        
        # Display current LTP values
        st.subheader("Current Option LTP")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**ATM-1 ({current_data['atm_strike']-50})**")
            st.write(f"CALL: ₹{current_data['atm_minus_call']:.2f}")
            st.write(f"PUT: ₹{current_data['atm_minus_put']:.2f}")
        
        with col2:
            st.write(f"**ATM ({current_data['atm_strike']})**")
            st.write(f"CALL: ₹{current_data['atm_call']:.2f}")
            st.write(f"PUT: ₹{current_data['atm_put']:.2f}")
        
        with col3:
            st.write(f"**ATM+1 ({current_data['atm_strike']+50})**")
            st.write(f"CALL: ₹{current_data['atm_plus_call']:.2f}")
            st.write(f"PUT: ₹{current_data['atm_plus_put']:.2f}")
        
        # Create charts if we have enough data
        if len(st.session_state.price_history['timestamps']) >= 10:
            st.subheader("SuperTrend Charts")
            
            # Create 2x3 grid for charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("CALL Options")
                
                # ATM Call
                chart = create_chart('atm_call', f'ATM CALL ({current_data["atm_strike"]})')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # ATM+1 Call
                chart = create_chart('atm_plus_call', f'ATM+1 CALL ({current_data["atm_strike"]+50})')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # ATM-1 Call
                chart = create_chart('atm_minus_call', f'ATM-1 CALL ({current_data["atm_strike"]-50})')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            with col2:
                st.subheader("PUT Options")
                
                # ATM Put
                chart = create_chart('atm_put', f'ATM PUT ({current_data["atm_strike"]})')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # ATM+1 Put
                chart = create_chart('atm_plus_put', f'ATM+1 PUT ({current_data["atm_strike"]+50})')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # ATM-1 Put
                chart = create_chart('atm_minus_put', f'ATM-1 PUT ({current_data["atm_strike"]-50})')
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
        else:
            st.info(f"Collecting data... ({len(st.session_state.price_history['timestamps'])}/10 points needed)")
    
    else:
        st.error("Failed to fetch option data")

if __name__ == "__main__":
    main()