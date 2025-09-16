import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta
import pytz

# Page config
st.set_page_config(
    page_title="Options SuperTrend Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
div[data-testid="metric-container"] { 
    background-color: #262730; 
    border: 1px solid #434651; 
    padding: 10px; 
    border-radius: 5px; 
}
</style>
""", unsafe_allow_html=True)

class DhanAPI:
    def __init__(self):
        self.base_url = "https://api.dhan.co/v2"
        self.client_id = st.secrets["DHAN_CLIENT_ID"]
        self.access_token = st.secrets["DHAN_ACCESS_TOKEN"]
        
        self.headers = {
            'access-token': self.access_token,
            'client-id': self.client_id,
            'Content-Type': 'application/json'
        }
        
        # Mapping of underlying symbols to security IDs
        self.underlyings = {
            "NIFTY": {"id": 13, "segment": "IDX_I"},
            "BANKNIFTY": {"id": 25, "segment": "IDX_I"}
        }
    
    def get_expiry_list(self, underlying):
        """Get available expiry dates for underlying"""
        underlying_info = self.underlyings.get(underlying)
        if not underlying_info:
            return []
        
        url = f"{self.base_url}/optionchain/expirylist"
        payload = {
            "UnderlyingScrip": underlying_info["id"],
            "UnderlyingSeg": underlying_info["segment"]
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            else:
                st.error(f"Expiry API Error: {response.status_code}")
                return []
        except Exception as e:
            st.error(f"Expiry API Exception: {str(e)}")
            return []
    
    def get_option_chain(self, underlying, expiry):
        """Get option chain for underlying and expiry"""
        underlying_info = self.underlyings.get(underlying)
        if not underlying_info:
            return None
        
        url = f"{self.base_url}/optionchain"
        payload = {
            "UnderlyingScrip": underlying_info["id"],
            "UnderlyingSeg": underlying_info["segment"],
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
            st.error(f"Option Chain Exception: {str(e)}")
            return None
    
    def get_historical_data(self, security_id, exchange_segment, instrument, from_date, to_date):
        """Get 1-minute historical intraday data"""
        url = f"{self.base_url}/charts/intraday"
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": "1",  # 1-minute intervals
            "fromDate": from_date,
            "toDate": to_date
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Historical Data API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Historical Data Exception: {str(e)}")
            return None

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period + 1:
        return []
    
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
        
        # Subsequent ATR values using exponential smoothing
        for i in range(period, len(tr)):
            atr_val = (atr[-1] * (period - 1) + tr[i]) / period
            atr.append(atr_val)
    
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate SuperTrend indicator"""
    if len(high) < period + 1:
        return [], []
    
    # Calculate HL2 and ATR
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    atr = calculate_atr(high, low, close, period)
    
    if not atr:
        return [], []
    
    # Initialize arrays
    upper_band = []
    lower_band = []
    supertrend = []
    trend = []
    
    # Pad ATR to match HL2 length
    atr_padded = [atr[0]] + atr  # Pad first value
    
    for i in range(len(hl2)):
        if i < len(atr_padded):
            ub = hl2[i] + multiplier * atr_padded[i]
            lb = hl2[i] - multiplier * atr_padded[i]
        else:
            ub = hl2[i] + multiplier * atr_padded[-1]
            lb = hl2[i] - multiplier * atr_padded[-1]
        
        if i == 0:
            upper_band.append(ub)
            lower_band.append(lb)
            supertrend.append(ub)
            trend.append(-1)  # Start with downtrend
        else:
            # Calculate final upper and lower bands
            if ub < upper_band[-1] or close[i-1] > upper_band[-1]:
                final_ub = ub
            else:
                final_ub = upper_band[-1]
                
            if lb > lower_band[-1] or close[i-1] < lower_band[-1]:
                final_lb = lb
            else:
                final_lb = lower_band[-1]
            
            upper_band.append(final_ub)
            lower_band.append(final_lb)
            
            # Determine trend
            if trend[-1] == 1 and close[i] <= final_lb:
                current_trend = -1
                current_st = final_ub
            elif trend[-1] == -1 and close[i] >= final_ub:
                current_trend = 1
                current_st = final_lb
            else:
                current_trend = trend[-1]
                if current_trend == 1:
                    current_st = final_lb
                else:
                    current_st = final_ub
            
            trend.append(current_trend)
            supertrend.append(current_st)
    
    return supertrend, trend

def create_option_data_from_underlying(underlying_data, strike, option_type, current_spot, current_ltp):
    """Create realistic option OHLCV data from underlying index data"""
    timestamps = underlying_data['timestamp']
    underlying_prices = {
        'open': underlying_data['open'],
        'high': underlying_data['high'], 
        'low': underlying_data['low'],
        'close': underlying_data['close']
    }
    
    option_data = {
        'timestamp': timestamps,
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': underlying_data.get('volume', [100] * len(timestamps))
    }
    
    for i in range(len(timestamps)):
        # Get underlying OHLC for this timeframe
        underlying_ohlc = {
            'open': underlying_prices['open'][i],
            'high': underlying_prices['high'][i],
            'low': underlying_prices['low'][i],
            'close': underlying_prices['close'][i]
        }
        
        # Calculate option prices for each OHLC point
        option_ohlc = {}
        for price_type, underlying_price in underlying_ohlc.items():
            
            # Calculate intrinsic value
            if option_type == "CE":
                intrinsic = max(0, underlying_price - strike)
                # Add time value based on moneyness and volatility
                moneyness = (underlying_price - strike) / strike
                time_value = max(0.5, abs(moneyness * 100) + 10)
            else:  # PE
                intrinsic = max(0, strike - underlying_price)
                moneyness = (strike - underlying_price) / strike
                time_value = max(0.5, abs(moneyness * 100) + 10)
            
            # Total option price
            option_price = intrinsic + time_value
            
            # Scale to make it realistic relative to current LTP
            if i == len(timestamps) - 1 and price_type == 'close' and current_ltp > 0:
                # Last close should match current LTP
                option_ohlc[price_type] = current_ltp
            else:
                # Scale other prices proportionally
                if current_ltp > 0:
                    scale_factor = current_ltp / (max(0.1, intrinsic + time_value))
                    option_price *= scale_factor
                option_ohlc[price_type] = max(0.05, option_price)
        
        # Ensure OHLC relationships are maintained
        option_ohlc['high'] = max(option_ohlc['open'], option_ohlc['high'], 
                                 option_ohlc['low'], option_ohlc['close'])
        option_ohlc['low'] = min(option_ohlc['open'], option_ohlc['high'], 
                                option_ohlc['low'], option_ohlc['close'])
        
        # Add to arrays
        option_data['open'].append(option_ohlc['open'])
        option_data['high'].append(option_ohlc['high'])
        option_data['low'].append(option_ohlc['low'])
        option_data['close'].append(option_ohlc['close'])
    
    return option_data

def get_trading_days_range(days_back=3):
    """Get date range for last N trading days (excluding weekends)"""
    ist = pytz.timezone('Asia/Kolkata')
    end_date = datetime.now(ist)
    
    # Go back enough days to ensure we get N trading days
    start_date = end_date - timedelta(days=days_back * 2)  # Buffer for weekends
    
    from_date = start_date.strftime('%Y-%m-%d 09:15:00')
    to_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    return from_date, to_date

def create_supertrend_chart(df, symbol):
    """Create candlestick chart with SuperTrend overlay"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol,
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='rgba(0,255,136,0.3)',
        decreasing_fillcolor='rgba(255,68,68,0.3)'
    ))
    
    # Add SuperTrend line
    if 'supertrend' in df.columns and 'trend' in df.columns:
        # Split SuperTrend into up and down segments
        st_up = df['supertrend'].where(df['trend'] == 1)
        st_down = df['supertrend'].where(df['trend'] == -1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=st_up,
            mode='lines',
            name='SuperTrend Up',
            line=dict(color='#00ff88', width=2),
            connectgaps=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=st_down,
            mode='lines',
            name='SuperTrend Down',
            line=dict(color='#ff4444', width=2),
            connectgaps=False
        ))
        
        # Add background color based on trend
        for i in range(len(df) - 1):
            color = 'rgba(0,255,136,0.1)' if df['trend'].iloc[i] == 1 else 'rgba(255,68,68,0.1)'
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                x1=df.index[i + 1],
                y0=df['low'].min() * 0.95,
                y1=df['high'].max() * 1.05,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0
            )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - 3 Days Intraday with SuperTrend",
        template='plotly_dark',
        height=600,
        xaxis_title="Time",
        yaxis_title="Price (₹)",
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    st.title("Options SuperTrend Analysis - DhanHQ API v2.0")
    
    # Initialize API
    api = DhanAPI()
    
    # Sidebar inputs
    st.sidebar.header("Configuration")
    
    # Underlying selection
    underlying = st.sidebar.selectbox(
        "Select Underlying",
        ["NIFTY", "BANKNIFTY"],
        index=0
    )
    
    # Get and display expiry dates
    with st.spinner("Fetching expiry dates..."):
        expiry_dates = api.get_expiry_list(underlying)
    
    if not expiry_dates:
        st.error("Failed to fetch expiry dates. Please check API credentials.")
        return
    
    expiry = st.sidebar.selectbox(
        "Select Expiry Date",
        expiry_dates,
        index=0 if expiry_dates else None
    )
    
    # Get option chain and extract strikes
    if expiry:
        with st.spinner("Fetching option chain..."):
            option_chain = api.get_option_chain(underlying, expiry)
        
        if option_chain and 'data' in option_chain:
            spot_price = option_chain['data'].get('last_price', 0)
            oc_data = option_chain['data'].get('oc', {})
            
            # Extract available strikes
            available_strikes = []
            for strike_key in oc_data.keys():
                try:
                    strike_price = int(float(strike_key.split('.')[0]))
                    available_strikes.append(strike_price)
                except:
                    continue
            
            available_strikes.sort()
            
            if available_strikes:
                # Strike selection
                selected_strike = st.sidebar.selectbox(
                    "Select Strike Price",
                    available_strikes,
                    index=len(available_strikes)//2 if available_strikes else 0
                )
                
                # Option type selection
                option_type = st.sidebar.radio(
                    "Select Option Type",
                    ["Call (CE)", "Put (PE)"],
                    index=0
                )
                option_type_short = "CE" if option_type == "Call (CE)" else "PE"
                
                # SuperTrend settings
                st.sidebar.subheader("SuperTrend Settings")
                st_period = st.sidebar.number_input(
                    "ATR Period",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=1
                )
                
                st_multiplier = st.sidebar.number_input(
                    "Multiplier",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.1
                )
                
                # Refresh button
                if st.sidebar.button("🔄 Refresh Chart", type="primary"):
                    st.rerun()
                
                # Main content
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(f"{underlying} Spot", f"₹{spot_price:.2f}")
                
                with col2:
                    # Get current option LTP
                    strike_key = f"{selected_strike}.000000"
                    current_ltp = 0
                    if strike_key in oc_data:
                        option_data = oc_data[strike_key].get(option_type_short.lower(), {})
                        current_ltp = option_data.get('last_price', 0)
                    
                    st.metric(f"{selected_strike} {option_type_short} LTP", f"₹{current_ltp:.2f}")
                
                with col3:
                    ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')
                    st.metric("Current Time", ist_time)
                
                with col4:
                    # Market status
                    ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                    market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
                    market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
                    market_status = "OPEN" if market_open <= ist_now <= market_close else "CLOSED"
                    st.metric("Market Status", market_status)
                
                # Fetch historical data
                st.subheader(f"📈 {selected_strike} {option_type_short} - Simulated Historical Data")
                
                # Since option-specific security IDs are not available from option chain,
                # we'll create realistic option data based on underlying movement
                with st.spinner("Generating option data from underlying movement..."):
                    from_date, to_date = get_trading_days_range(3)
                    
                    # Get underlying historical data first
                    underlying_info = api.underlyings[underlying]
                    underlying_hist = api.get_historical_data(
                        underlying_info["id"],
                        underlying_info["segment"],
                        "INDEX",
                        from_date,
                        to_date
                    )
                    
                    # Create option data from underlying data
                    if underlying_hist and 'timestamp' in underlying_hist:
                        hist_data = create_option_data_from_underlying(
                            underlying_hist, selected_strike, option_type_short, spot_price, current_ltp
                        )
                    else:
                        hist_data = None
                
                if hist_data and 'timestamp' in hist_data:
                    # Create DataFrame
                    df = pd.DataFrame({
                        'timestamp': hist_data['timestamp'],
                        'open': hist_data['open'],
                        'high': hist_data['high'],
                        'low': hist_data['low'],
                        'close': hist_data['close'],
                        'volume': hist_data.get('volume', [100] * len(hist_data['timestamp']))
                    })
                    
                    # Convert timestamp to datetime
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('datetime', inplace=True)
                    
                    # Convert to IST timezone
                    ist = pytz.timezone('Asia/Kolkata')
                    df.index = df.index.tz_localize('UTC').tz_convert(ist)
                    
                    # Filter for trading hours only (9:15 AM to 3:30 PM IST)
                    df = df.between_time('09:15', '15:30')
                    
                    # Remove any invalid data
                    df = df.dropna()
                    df = df[df['close'] > 0]  # Remove zero prices
                    
                    if len(df) > st_period + 5:  # Ensure we have enough data
                        # Calculate SuperTrend
                        supertrend_values, trend_values = calculate_supertrend(
                            df['high'].tolist(),
                            df['low'].tolist(), 
                            df['close'].tolist(),
                            period=st_period,
                            multiplier=st_multiplier
                        )
                        
                        if supertrend_values and trend_values and len(supertrend_values) > 0:
                            # Pad arrays to match DataFrame length
                            if len(supertrend_values) < len(df):
                                pad_length = len(df) - len(supertrend_values)
                                supertrend_values = [None] * pad_length + supertrend_values
                                trend_values = [None] * pad_length + trend_values
                            
                            df['supertrend'] = supertrend_values[:len(df)]
                            df['trend'] = trend_values[:len(df)]
                            
                            # Create and display chart
                            symbol = f"{selected_strike} {option_type_short}"
                            fig = create_supertrend_chart(df, symbol)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display latest trend (only if we have valid trend data)
                            valid_trends = [t for t in trend_values if t is not None]
                            if valid_trends:
                                latest_trend = valid_trends[-1]
                                trend_text = "Uptrend" if latest_trend == 1 else "Downtrend"
                                trend_color = "🟢" if latest_trend == 1 else "🔴"
                                
                                col_trend1, col_trend2 = st.columns(2)
                                with col_trend1:
                                    st.metric("Current Trend Direction", f"{trend_color} {trend_text}")
                                
                                with col_trend2:
                                    valid_st_values = [s for s in supertrend_values if s is not None]
                                    if valid_st_values:
                                        latest_st_value = valid_st_values[-1]
                                        st.metric("SuperTrend Value", f"₹{latest_st_value:.2f}")
                            
                            # Show data summary
                            st.success(f"📊 Generated {len(df)} option data points from {df.index[0].strftime('%d-%m %H:%M')} to {df.index[-1].strftime('%d-%m %H:%M')} IST")
                            
                            # Show latest values
                            with st.expander("Latest OHLC Values"):
                                latest = df.iloc[-1]
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Open", f"₹{latest['open']:.2f}")
                                with col2:
                                    st.metric("High", f"₹{latest['high']:.2f}")
                                with col3:
                                    st.metric("Low", f"₹{latest['low']:.2f}")
                                with col4:
                                    st.metric("Close", f"₹{latest['close']:.2f}")
                        
                        else:
                            st.error("Failed to calculate SuperTrend. Please try different parameters.")
                    
                    else:
                        st.warning(f"Insufficient data points. Need at least {st_period + 5}, got {len(df)}. Try reducing SuperTrend period.")
                
                else:
                    st.error("Failed to generate option data from underlying. Please check API connection.")
            
            else:
                st.error("No strikes available for the selected expiry.")
        
        else:
            st.error("Failed to fetch option chain data.")
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    **Note:** This application uses DhanHQ API v2.0 for real-time market data. 
    Historical data is fetched using the `/charts/intraday` endpoint with 1-minute intervals.
    
    **Features:**
    - Real-time option chain data
    - 3-day historical intraday data
    - SuperTrend technical indicator
    - Candlestick charts with trend analysis
    """)

if __name__ == "__main__":
    main()
