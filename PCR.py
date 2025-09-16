import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import pytz
import threading
from concurrent.futures import ThreadPoolExecutor

# Set page config
st.set_page_config(
    page_title="Nifty Options Trading Chart",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #131722;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #434651;
    }
    .status-positive {
        color: #4CAF50;
    }
    .status-negative {
        color: #F44336;
    }
    .strike-info {
        background-color: #2a2e39;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .time-display {
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        color: #00ff88;
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class DhanAPI:
    def __init__(self):
        self.base_url = "https://api.dhan.co/v2"
        self.access_token = st.secrets.get("DHAN_ACCESS_TOKEN", "")
        self.client_id = st.secrets.get("DHAN_CLIENT_ID", "")
        
        if not self.access_token or not self.client_id:
            st.error("Please configure DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID in Streamlit secrets")
            st.stop()
        
        self.headers = {
            'access-token': self.access_token,
            'client-id': self.client_id,
            'Content-Type': 'application/json'
        }
    
    def get_historical_data(self, security_id, exchange_segment, instrument, from_date, to_date, interval="1"):
        """Get historical intraday data"""
        url = f"{self.base_url}/charts/intraday"
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": True,
            "fromDate": from_date,
            "toDate": to_date
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Historical Data API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Historical Data Error: {str(e)}")
            return None
    
    def get_market_quote(self, instruments):
        """Get market quote for multiple instruments"""
        url = f"{self.base_url}/marketfeed/quote"
        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None
    
    def get_option_chain(self, underlying_scrip, underlying_seg, expiry):
        """Get option chain data"""
        url = f"{self.base_url}/optionchain"
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg,
            "Expiry": expiry
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Option Chain API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Option Chain Error: {str(e)}")
            return None
    
    def get_expiry_list(self, underlying_scrip, underlying_seg):
        """Get available expiry dates"""
        url = f"{self.base_url}/optionchain/expirylist"
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            return None

class SuperTrendCalculator:
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Calculate Average True Range"""
        tr = []
        for i in range(1, len(high)):
            tr_val = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr.append(tr_val)
        
        atr = []
        if len(tr) >= period:
            # First ATR
            atr.append(sum(tr[:period]) / period)
            
            # Subsequent ATR values
            for i in range(period, len(tr)):
                atr_val = (atr[-1] * (period - 1) + tr[i]) / period
                atr.append(atr_val)
        
        return atr
    
    @staticmethod
    def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
        """Calculate SuperTrend indicator"""
        if len(high) < period:
            return [], []
        
        hl2 = [(h + l) / 2 for h, l in zip(high, low)]
        atr = SuperTrendCalculator.calculate_atr(high, low, close, period)
        
        upper_band = []
        lower_band = []
        supertrend = []
        trend_direction = []
        
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
                # Update bands
                ub = ub if ub < upper_band[-1] or close[i-1] > upper_band[-1] else upper_band[-1]
                lb = lb if lb > lower_band[-1] or close[i-1] < lower_band[-1] else lower_band[-1]
                
                upper_band.append(ub)
                lower_band.append(lb)
                
                # Determine trend
                if trend_direction[-1] == 1 and close[i] <= lb:
                    trend_direction.append(-1)
                elif trend_direction[-1] == -1 and close[i] >= ub:
                    trend_direction.append(1)
                else:
                    trend_direction.append(trend_direction[-1])
                
                st_val = lb if trend_direction[-1] == 1 else ub
                supertrend.append(st_val)
        
        return supertrend, trend_direction

class OptionsDataManager:
    def __init__(self):
        if 'options_data' not in st.session_state:
            st.session_state.options_data = {
                'timestamps': [],
                'ist_timestamps': [],
                'spot_prices': [],
                'strikes_data': {},
                'supertrend_up': [],
                'supertrend_down': [],
                'historical_loaded': False
            }
        
        if 'selected_strikes' not in st.session_state:
            st.session_state.selected_strikes = {
                'call': ['ATM-1', 'ATM', 'ATM+1'],
                'put': []
            }
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def get_ist_time(self):
        """Get current IST time"""
        ist = pytz.timezone('Asia/Kolkata')
        return datetime.now(ist)
    
    def get_atm_strike(self, spot_price):
        """Calculate ATM strike price (nearest 50)"""
        return round(spot_price / 50) * 50
    
    def get_strike_price(self, atm_strike, reference):
        """Convert ATM reference to actual strike price"""
        if reference == 'ATM':
            return atm_strike
        elif reference.startswith('ATM+'):
            offset = int(reference.replace('ATM+', ''))
            return atm_strike + (offset * 50)
        elif reference.startswith('ATM-'):
            offset = int(reference.replace('ATM-', ''))
            return atm_strike - (offset * 50)
        return atm_strike
    
    def load_historical_data(self, api, days_back=5):
        """Load historical data for context"""
        try:
            if st.session_state.options_data['historical_loaded']:
                return True
            
            ist = pytz.timezone('Asia/Kolkata')
            end_date = datetime.now(ist)
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d 09:15:00')
            to_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Get Nifty historical data for reference
            historical_data = api.get_historical_data(
                security_id=13,  # Nifty 50 index
                exchange_segment="IDX_I",
                instrument="INDEX",
                from_date=from_date,
                to_date=to_date,
                interval="5"  # 5-minute intervals
            )
            
            if historical_data and 'timestamp' in historical_data:
                timestamps = historical_data['timestamp']
                closes = historical_data['close']
                
                # Convert timestamps to IST
                for i, ts in enumerate(timestamps[-20:]):  # Last 20 points
                    dt = datetime.fromtimestamp(ts, ist)
                    time_str = dt.strftime('%H:%M:%S')
                    ist_str = dt.strftime('%d-%m %H:%M')
                    
                    st.session_state.options_data['timestamps'].append(time_str)
                    st.session_state.options_data['ist_timestamps'].append(ist_str)
                    st.session_state.options_data['spot_prices'].append(closes[i] if i < len(closes) else 25000)
                
                st.session_state.options_data['historical_loaded'] = True
                return True
            
        except Exception as e:
            st.error(f"Historical data loading error: {str(e)}")
            return False
        
        return False
    
    def update_data(self, api, expiry_date):
        """Update options data from API"""
        try:
            # Get option chain data
            option_chain = api.get_option_chain(13, "IDX_I", expiry_date)
            
            if not option_chain or 'data' not in option_chain:
                return False
            
            ist = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(ist)
            time_str = current_time.strftime('%H:%M:%S')
            ist_str = current_time.strftime('%d-%m %H:%M')
            
            spot_price = option_chain['data'].get('last_price', 25000)
            atm_strike = self.get_atm_strike(spot_price)
            
            # Add timestamp and spot price
            st.session_state.options_data['timestamps'].append(time_str)
            st.session_state.options_data['ist_timestamps'].append(ist_str)
            st.session_state.options_data['spot_prices'].append(spot_price)
            
            # Process selected strikes
            oc_data = option_chain['data'].get('oc', {})
            
            for strike_ref in st.session_state.selected_strikes['call']:
                strike_price = self.get_strike_price(atm_strike, strike_ref)
                strike_key = f"{strike_price}.000000"
                key = f"{strike_ref}_CE"
                
                if strike_key in oc_data and 'ce' in oc_data[strike_key]:
                    price = oc_data[strike_key]['ce'].get('last_price', 0)
                    if key not in st.session_state.options_data['strikes_data']:
                        st.session_state.options_data['strikes_data'][key] = []
                    st.session_state.options_data['strikes_data'][key].append(price)
            
            for strike_ref in st.session_state.selected_strikes['put']:
                strike_price = self.get_strike_price(atm_strike, strike_ref)
                strike_key = f"{strike_price}.000000"
                key = f"{strike_ref}_PE"
                
                if strike_key in oc_data and 'pe' in oc_data[strike_key]:
                    price = oc_data[strike_key]['pe'].get('last_price', 0)
                    if key not in st.session_state.options_data['strikes_data']:
                        st.session_state.options_data['strikes_data'][key] = []
                    st.session_state.options_data['strikes_data'][key].append(price)
            
            # Limit data points to last 200
            max_points = 200
            if len(st.session_state.options_data['timestamps']) > max_points:
                st.session_state.options_data['timestamps'] = st.session_state.options_data['timestamps'][-max_points:]
                st.session_state.options_data['ist_timestamps'] = st.session_state.options_data['ist_timestamps'][-max_points:]
                st.session_state.options_data['spot_prices'] = st.session_state.options_data['spot_prices'][-max_points:]
                
                for key in st.session_state.options_data['strikes_data']:
                    if len(st.session_state.options_data['strikes_data'][key]) > max_points:
                        st.session_state.options_data['strikes_data'][key] = st.session_state.options_data['strikes_data'][key][-max_points:]
            
            # Calculate SuperTrend on ATM call option
            atm_call_key = 'ATM_CE'
            if (atm_call_key in st.session_state.options_data['strikes_data'] and 
                len(st.session_state.options_data['strikes_data'][atm_call_key]) >= 10):
                
                prices = st.session_state.options_data['strikes_data'][atm_call_key]
                high = [p * 1.02 for p in prices]  # Simulate high
                low = [p * 0.98 for p in prices]   # Simulate low
                close = prices
                
                period = st.session_state.get('st_period', 10)
                multiplier = st.session_state.get('st_multiplier', 3.0)
                
                supertrend, trend_direction = SuperTrendCalculator.calculate_supertrend(
                    high, low, close, period, multiplier
                )
                
                # Split into up and down arrays
                st.session_state.options_data['supertrend_up'] = []
                st.session_state.options_data['supertrend_down'] = []
                
                for i, (st_val, direction) in enumerate(zip(supertrend, trend_direction)):
                    if direction == 1:
                        st.session_state.options_data['supertrend_up'].append(st_val)
                        st.session_state.options_data['supertrend_down'].append(None)
                    else:
                        st.session_state.options_data['supertrend_up'].append(None)
                        st.session_state.options_data['supertrend_down'].append(st_val)
            
            # Update last update time
            st.session_state.last_update = current_time
            
            return True
            
        except Exception as e:
            st.error(f"Data update error: {str(e)}")
            return False

def create_chart():
    """Create the main chart using Plotly"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["Nifty Options LTP with SuperTrend (IST)"],
        vertical_spacing=0.05
    )
    
    # Color palette for strikes
    colors = ['#FF4444', '#FF8800', '#2196F3', '#4CAF50', '#9C27B0', 
              '#E91E63', '#00BCD4', '#FF9800', '#795548', '#607D8B']
    
    data = st.session_state.options_data
    timestamps = data['ist_timestamps']  # Use IST timestamps
    
    if not timestamps:
        fig.add_annotation(
            text="Loading data... Please wait",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="white")
        )
    else:
        color_idx = 0
        
        # Add selected strikes
        all_strikes = (st.session_state.selected_strikes['call'] + 
                      st.session_state.selected_strikes['put'])
        
        for strike_ref in all_strikes:
            option_type = 'CE' if strike_ref in st.session_state.selected_strikes['call'] else 'PE'
            key = f"{strike_ref}_{option_type}"
            
            if key in data['strikes_data'] and data['strikes_data'][key]:
                spot_price = data['spot_prices'][-1] if data['spot_prices'] else 25000
                atm_strike = round(spot_price / 50) * 50
                
                if strike_ref == 'ATM':
                    strike_price = atm_strike
                elif strike_ref.startswith('ATM+'):
                    offset = int(strike_ref.replace('ATM+', ''))
                    strike_price = atm_strike + (offset * 50)
                elif strike_ref.startswith('ATM-'):
                    offset = int(strike_ref.replace('ATM-', ''))
                    strike_price = atm_strike - (offset * 50)
                else:
                    strike_price = atm_strike
                
                # Ensure data length matches timestamps
                strike_data = data['strikes_data'][key]
                if len(strike_data) < len(timestamps):
                    # Pad with None values if needed
                    strike_data = [None] * (len(timestamps) - len(strike_data)) + strike_data
                elif len(strike_data) > len(timestamps):
                    # Trim if too long
                    strike_data = strike_data[-len(timestamps):]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=strike_data,
                        mode='lines+markers',
                        name=f"{strike_price} {option_type}",
                        line=dict(color=colors[color_idx % len(colors)], width=2),
                        marker=dict(size=4),
                        hovertemplate=f"{strike_price} {option_type}<br>Price: ₹%{{y:.2f}}<br>Time: %{{x}}<extra></extra>",
                        connectgaps=True
                    )
                )
                color_idx += 1
        
        # Add SuperTrend
        if data['supertrend_up'] and len(data['supertrend_up']) > 0:
            # Align SuperTrend data with timestamps
            st_up = data['supertrend_up']
            st_down = data['supertrend_down']
            
            if len(st_up) < len(timestamps):
                st_up = [None] * (len(timestamps) - len(st_up)) + st_up
                st_down = [None] * (len(timestamps) - len(st_down)) + st_down
            elif len(st_up) > len(timestamps):
                st_up = st_up[-len(timestamps):]
                st_down = st_down[-len(timestamps):]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=st_up,
                    mode='lines',
                    name='SuperTrend Up',
                    line=dict(color='#E91E63', width=3),
                    connectgaps=False,
                    hovertemplate="SuperTrend Up<br>Value: %{y:.2f}<br>Time: %{x}<extra></extra>"
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=st_down,
                    mode='lines',
                    name='SuperTrend Down',
                    line=dict(color='#9C27B0', width=3),
                    connectgaps=False,
                    hovertemplate="SuperTrend Down<br>Value: %{y:.2f}<br>Time: %{x}<extra></extra>"
                )
            )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title="Time (IST)",
            showgrid=True,
            gridcolor='#2a2e39',
            type='category'  # Treat x-axis as categorical for proper spacing
        ),
        yaxis=dict(
            title="Option Price (₹)",
            showgrid=True,
            gridcolor='#2a2e39'
        ),
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        hovermode='x unified'
    )
    
    return fig

def auto_refresh_data(api, data_manager, expiry_date, interval_seconds):
    """Background function for auto-refreshing data"""
    if 'auto_refresh_active' not in st.session_state:
        st.session_state.auto_refresh_active = False
    
    if st.session_state.auto_refresh_active:
        success = data_manager.update_data(api, expiry_date)
        if success:
            st.session_state.refresh_counter = st.session_state.get('refresh_counter', 0) + 1

def main():
    st.markdown('<h1 class="main-header">📈 Nifty Options Trading Chart with IST Time</h1>', unsafe_allow_html=True)
    
    # Initialize components
    api = DhanAPI()
    data_manager = OptionsDataManager()
    
    # Initialize session state for auto-refresh
    if 'auto_refresh_active' not in st.session_state:
        st.session_state.auto_refresh_active = False
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    # Sidebar controls
    st.sidebar.header("⚙️ Chart Controls")
    
    # Current IST time display
    current_ist = data_manager.get_ist_time()
    st.sidebar.markdown(f"""
    <div class="time-display">
        🕒 IST: {current_ist.strftime('%d-%m-%Y %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)
    
    # SuperTrend parameters
    st.sidebar.subheader("SuperTrend Settings")
    st.session_state.st_period = st.sidebar.selectbox(
        "Period", [10, 14, 21], index=0, key="st_period_select"
    )
    st.session_state.st_multiplier = st.sidebar.selectbox(
        "Multiplier", [2.0, 3.0, 4.0], index=1, key="st_multiplier_select"
    )
    
    # Time interval for auto-refresh
    st.sidebar.subheader("Auto Refresh")
    time_interval = st.sidebar.selectbox(
        "Update Interval", 
        ["5 seconds", "10 seconds", "15 seconds", "30 seconds"], 
        index=1
    )
    
    # Auto refresh toggle
    auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=False)
    st.session_state.auto_refresh_active = auto_refresh
    
    # Strike selection
    st.sidebar.subheader("📊 Strike Selection")
    
    strikes_available = ['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']
    
    st.sidebar.write("**Call Options (CE)**")
    selected_calls = st.sidebar.multiselect(
        "Select Call Strikes",
        strikes_available,
        default=st.session_state.selected_strikes['call'],
        key="call_strikes"
    )
    
    st.sidebar.write("**Put Options (PE)**")
    selected_puts = st.sidebar.multiselect(
        "Select Put Strikes",
        strikes_available,
        default=st.session_state.selected_strikes['put'],
        key="put_strikes"
    )
    
    # Update selection
    st.session_state.selected_strikes['call'] = selected_calls
    st.session_state.selected_strikes['put'] = selected_puts
    
    # Get expiry dates
    expiry_data = api.get_expiry_list(13, "IDX_I")
    if expiry_data and 'data' in expiry_data:
        expiry_dates = expiry_data['data']
        selected_expiry = st.sidebar.selectbox(
            "Expiry Date", expiry_dates, index=0 if expiry_dates else None
        )
    else:
        st.sidebar.error("Could not fetch expiry dates")
        selected_expiry = "2024-10-31"  # Fallback
    
    # Control buttons
    st.sidebar.subheader("🎯 Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            with st.spinner("Fetching latest data..."):
                success = data_manager.update_data(api, selected_expiry)
                if success:
                    st.sidebar.success("Data updated!")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to update data")
    
    with col2:
        if st.button("📈 Load Historical", use_container_width=True):
            with st.spinner("Loading historical data..."):
                success = data_manager.load_historical_data(api, days_back=2)
                if success:
                    st.sidebar.success("Historical data loaded!")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to load historical data")
    
    if st.sidebar.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.options_data = {
            'timestamps': [],
            'ist_timestamps': [],
            'spot_prices': [],
            'strikes_data': {},
            'supertrend_up': [],
            'supertrend_down': [],
            'historical_loaded': False
        }
        st.sidebar.success("Data cleared!")
        st.rerun()
    
    # Main content area
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Display current metrics
    data = st.session_state.options_data
    
    if data['spot_prices']:
        current_spot = data['spot_prices'][-1]
        atm_strike = round(current_spot / 50) * 50
        
        with col1:
            st.metric("Nifty Spot", f"₹{current_spot:.2f}")
        
        with col2:
            st.metric("ATM Strike", f"{atm_strike}")
        
        with col3:
            if st.session_state.last_update:
                last_update_str = st.session_state.last_update.strftime('%H:%M:%S')
                st.metric("Last Update", last_update_str)
            else:
                st.metric("Last Update", "No data")
        
        with col4:
            total_selected = len(selected_calls) + len(selected_puts)
            st.metric("Active Options", f"{total_selected}")
            st.caption(f"{len(selected_calls)} CE, {len(selected_puts)} PE")
        
        with col5:
            data_points = len(data['timestamps'])
            st.metric("Data Points", f"{data_points}")
            if auto_refresh:
                st.caption("🟢 Auto-refresh ON")
            else:
                st.caption("🔴 Auto-refresh OFF")
        
        # SuperTrend signal
        if data['supertrend_up'] and data['supertrend_down']:
            last_up = data['supertrend_up'][-1] if data['supertrend_up'] else None
            last_down = data['supertrend_down'][-1] if data['supertrend_down'] else None
            
            if last_up is not None:
                signal = "🟢 BULLISH TREND"
                signal_class = "status-positive"
            elif last_down is not None:
                signal = "🔴 BEARISH TREND"
                signal_class = "status-negative"
            else:
                signal = "⚪ NEUTRAL"
                signal_class = ""
            
            st.markdown(f'<p style="text-align: center; font-size: 1.2rem;" class="{signal_class}"><strong>SuperTrend Signal: {signal}</strong></p>', 
                       unsafe_allow_html=True)
    
    # Main chart
    chart_placeholder = st.empty()
    
    with chart_placeholder.container():
        fig = create_chart()
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto refresh logic with proper timing
    if auto_refresh:
        interval_map = {
            "5 seconds": 5,
            "10 seconds": 10,
            "15 seconds": 15,
            "30 seconds": 30
        }
        
        refresh_interval = interval_map.get(time_interval, 10)
        
        # Initialize refresh timestamp
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()
        
        current_time = time.time()
        time_since_last_refresh = current_time - st.session_state.last_refresh_time
        
        # Auto refresh placeholder
        refresh_placeholder = st.empty()
        
        if time_since_last_refresh >= refresh_interval:
            with refresh_placeholder.container():
                with st.spinner(f"Auto-refreshing data... (Every {time_interval})"):
                    success = data_manager.update_data(api, selected_expiry)
                    if success:
                        st.session_state.last_refresh_time = current_time
                        st.session_state.refresh_counter += 1
                        st.success(f"Data updated automatically at {current_ist.strftime('%H:%M:%S')}")
                        time.sleep(1)  # Brief pause to show success message
                        st.rerun()
        else:
            # Show countdown to next refresh
            time_remaining = refresh_interval - time_since_last_refresh
            refresh_placeholder.info(f"Next auto-refresh in {time_remaining:.0f} seconds...")
    
    # Data summary section
    st.markdown("---")
    
    if data['timestamps']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Strike Prices")
            if data['spot_prices']:
                current_spot = data['spot_prices'][-1]
                atm_strike = round(current_spot / 50) * 50
                
                strike_info = []
                for i in range(-2, 3):
                    strike = atm_strike + (i * 50)
                    if i == 0:
                        strike_info.append(f"**{strike} (ATM)**")
                    elif i > 0:
                        strike_info.append(f"{strike} (ATM+{i})")
                    else:
                        strike_info.append(f"{strike} (ATM{i})")
                
                for info in strike_info:
                    st.write(f"• {info}")
        
        with col2:
            st.subheader("Session Statistics")
            st.write(f"• Total Updates: {st.session_state.refresh_counter}")
            st.write(f"• Data Points: {len(data['timestamps'])}")
            st.write(f"• Historical Loaded: {'Yes' if data.get('historical_loaded', False) else 'No'}")
            
            if data['timestamps']:
                first_time = data['ist_timestamps'][0] if data['ist_timestamps'] else "N/A"
                last_time = data['ist_timestamps'][-1] if data['ist_timestamps'] else "N/A"
                st.write(f"• Time Range: {first_time} to {last_time}")
    
    # Market status and timing info
    ist_now = data_manager.get_ist_time()
    market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if market_open <= ist_now <= market_close:
        market_status = "🟢 Market Open"
        status_color = "status-positive"
    else:
        market_status = "🔴 Market Closed"
        status_color = "status-negative"
    
    st.markdown(f'<p class="{status_color}"><strong>{market_status}</strong> | Market Hours: 09:15 - 15:30 IST</p>', 
               unsafe_allow_html=True)
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    1. Configure DhanHQ API credentials in Streamlit secrets
    2. Select desired strikes from sidebar (ATM ±2 range available)
    3. Enable auto-refresh for real-time updates during market hours
    4. Load historical data for context using the button
    5. SuperTrend indicator shows trend direction based on ATM Call option
    
    **Features:**
    - Real-time IST timestamps
    - Auto-refresh with configurable intervals
    - Historical data loading (last 2 days)
    - SuperTrend technical indicator
    - Professional dark theme optimized for trading
    
    **Note:** Data updates are subject to DhanHQ API rate limits (Option Chain: 1 request per 3 seconds)
    """)
    
    # Force refresh every few seconds when auto-refresh is enabled
    if auto_refresh:
        time.sleep(1)  # Small delay to prevent excessive CPU usage
        st.rerun()

if __name__ == "__main__":
    main()
