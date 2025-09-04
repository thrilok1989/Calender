import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from pytz import timezone
import io
import json

# Streamlit Configuration
st.set_page_config(
    page_title="DhanHQ Trading Dashboard with Volume Footprint",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()
if 'live_data' not in st.session_state:
    st.session_state.live_data = pd.DataFrame()
if 'footprint_data' not in st.session_state:
    st.session_state.footprint_data = pd.DataFrame()
if 'api_call_count' not in st.session_state:
    st.session_state.api_call_count = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# DhanHQ Configuration
DHAN_API_BASE_URL = "https://api.dhan.co/v2"

def get_dhan_config():
    """Get DhanHQ credentials from Streamlit secrets"""
    try:
        return {
            "client_id": st.secrets.get("DHAN_CLIENT_ID"),
            "access_token": st.secrets.get("DHAN_ACCESS_TOKEN")
        }
    except:
        return {"client_id": None, "access_token": None}

dhan_config = get_dhan_config()

def get_dhan_headers():
    """Get DhanHQ API headers"""
    if not dhan_config["client_id"] or not dhan_config["access_token"]:
        return None
    
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": dhan_config["access_token"],
        "client-id": dhan_config["client_id"]
    }

def make_api_request(url, headers, payload, timeout=15):
    """Make API request with proper error handling"""
    try:
        st.session_state.api_call_count += 1
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        if response.status_code == 429:
            st.warning("Rate limit reached. Waiting 3 seconds...")
            time.sleep(3)
            return None
        elif response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def get_historical_data(security_id, exchange_segment, instrument_type, from_date, to_date, interval="1"):
    """Fetch historical data from DhanHQ API"""
    headers = get_dhan_headers()
    if not headers:
        st.error("DhanHQ credentials not configured")
        return pd.DataFrame()
    
    try:
        # For intraday data
        if interval in ["1", "5", "15", "25", "60"]:
            url = f"{DHAN_API_BASE_URL}/charts/intraday"
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exchange_segment,
                "instrument": instrument_type,
                "interval": interval,
                "fromDate": from_date.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": to_date.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            # For daily data
            url = f"{DHAN_API_BASE_URL}/charts/historical"
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exchange_segment,
                "instrument": instrument_type,
                "fromDate": from_date.strftime("%Y-%m-%d"),
                "toDate": to_date.strftime("%Y-%m-%d")
            }
        
        st.info(f"Fetching historical data for {security_id} from {exchange_segment}...")
        data = make_api_request(url, headers, payload)
        
        if data and 'open' in data:
            df = pd.DataFrame({
                'timestamp': data['timestamp'],
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            })
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            df = df.sort_values('datetime').reset_index(drop=True)
            
            st.success(f"Fetched {len(df)} candles successfully")
            return df
        else:
            st.warning("No data received from API")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error in historical data fetch: {str(e)}")
        return pd.DataFrame()

def get_live_data(security_id, exchange_segment):
    """Get live market data"""
    headers = get_dhan_headers()
    if not headers:
        return None
    
    try:
        payload = {
            exchange_segment: [int(security_id)]
        }
        
        data = make_api_request(f"{DHAN_API_BASE_URL}/marketfeed/quote", headers, payload)
        
        if data and 'data' in data and exchange_segment in data['data']:
            security_data = data['data'][exchange_segment].get(str(security_id), {})
            
            if security_data:
                return {
                    'last_price': security_data.get('last_price', 0),
                    'volume': security_data.get('volume', 0),
                    'high': security_data.get('ohlc', {}).get('high', 0),
                    'low': security_data.get('ohlc', {}).get('low', 0),
                    'open': security_data.get('ohlc', {}).get('open', 0),
                    'close': security_data.get('ohlc', {}).get('close', 0),
                    'timestamp': datetime.now(timezone('Asia/Kolkata'))
                }
        return None
    except Exception as e:
        st.error(f"Live data error: {str(e)}")
        return None

class VolumeFootprintIndicator:
    """Volume Footprint Analysis with proper DataFrame handling"""
    
    def __init__(self, bins=20):
        self.bins = bins
        self.footprint_data = pd.DataFrame()
    
    def safe_dataframe_check(self, df):
        """Safely check if DataFrame is empty or None"""
        if df is None:
            return True
        if isinstance(df, pd.DataFrame) and df.empty:
            return True
        return False
    
    def calculate_volume_footprint(self, df, bins=None):
        """Calculate volume footprint from OHLCV data"""
        if self.safe_dataframe_check(df):
            return pd.DataFrame()
        
        bins = bins or self.bins
        
        try:
            # Create price levels based on high-low range
            df = df.copy()
            df['price_range'] = df['high'] - df['low']
            df['volume_per_tick'] = df['volume'] / np.maximum(df['price_range'], 0.01)
            
            # Create footprint data
            footprint_rows = []
            
            for idx, row in df.iterrows():
                if row['price_range'] > 0:
                    price_levels = np.linspace(row['low'], row['high'], bins)
                    volume_per_level = row['volume'] / bins
                    
                    for price in price_levels:
                        footprint_rows.append({
                            'datetime': row['datetime'],
                            'price': price,
                            'volume': volume_per_level,
                            'is_buy': price >= row['close'],  # Simplified buy/sell logic
                            'session_volume': row['volume']
                        })
            
            footprint_df = pd.DataFrame(footprint_rows)
            
            if not footprint_df.empty:
                # Aggregate by price levels
                footprint_agg = footprint_df.groupby(['datetime', 'price']).agg({
                    'volume': 'sum',
                    'is_buy': 'mean',
                    'session_volume': 'first'
                }).reset_index()
                
                return footprint_agg
            
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Volume footprint calculation error: {str(e)}")
            return pd.DataFrame()
    
    def detect_htf_change(self, current_data, previous_data):
        """Detect higher timeframe changes - fixed DataFrame boolean evaluation"""
        # Fix the boolean evaluation issue
        if self.safe_dataframe_check(current_data) or self.safe_dataframe_check(previous_data):
            return False
        
        try:
            if len(current_data) == 0 or len(previous_data) == 0:
                return False
            
            current_high = current_data['high'].max()
            current_low = current_data['low'].min()
            prev_high = previous_data['high'].max()
            prev_low = previous_data['low'].min()
            
            # Detect significant price level breaks
            range_expansion = (current_high - current_low) > 1.5 * (prev_high - prev_low)
            volume_spike = current_data['volume'].sum() > 2 * previous_data['volume'].mean()
            
            return range_expansion and volume_spike
        
        except Exception as e:
            st.error(f"HTF change detection error: {str(e)}")
            return False
    
    def update_volume_footprint(self, data_df):
        """Update volume footprint with proper error handling"""
        if self.safe_dataframe_check(data_df):
            return None, None
        
        try:
            # Calculate current session data
            current_session = data_df.tail(50)  # Last 50 candles
            previous_session = data_df.iloc[-100:-50] if len(data_df) > 100 else pd.DataFrame()
            
            # Check for higher timeframe changes
            htf_change = self.detect_htf_change(current_session, previous_session)
            
            # Calculate volume footprint
            footprint = self.calculate_volume_footprint(current_session)
            self.footprint_data = footprint
            
            if not self.safe_dataframe_check(footprint):
                session_high = current_session['high'].max()
                session_low = current_session['low'].min()
                
                return session_high, session_low
            
            return None, None
            
        except Exception as e:
            st.error(f"Volume footprint update error: {str(e)}")
            return None, None

def create_candlestick_chart(df, title="Price Chart"):
    """Create candlestick chart with volume"""
    if df.empty:
        st.warning("No data to display")
        return None
    
    # Create subplots
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        height=600,
        showlegend=True
    )
    
    return fig

def create_volume_footprint_chart(footprint_data, title="Volume Footprint"):
    """Create volume footprint visualization"""
    if footprint_data.empty:
        st.warning("No footprint data available")
        return None
    
    try:
        fig = go.Figure()
        
        # Group data by price level
        price_levels = footprint_data.groupby('price').agg({
            'volume': 'sum',
            'is_buy': 'mean'
        }).reset_index()
        
        # Create buy/sell volume bars
        buy_volume = price_levels['volume'] * price_levels['is_buy']
        sell_volume = price_levels['volume'] * (1 - price_levels['is_buy'])
        
        # Add buy volume (green)
        fig.add_trace(go.Bar(
            y=price_levels['price'],
            x=buy_volume,
            orientation='h',
            name='Buy Volume',
            marker_color='green',
            opacity=0.7
        ))
        
        # Add sell volume (red)
        fig.add_trace(go.Bar(
            y=price_levels['price'],
            x=-sell_volume,  # Negative for left side
            orientation='h',
            name='Sell Volume',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Volume",
            yaxis_title="Price Level",
            template="plotly_white",
            height=400,
            barmode='relative'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Footprint chart error: {str(e)}")
        return None

class TradingDashboard:
    """Main Trading Dashboard Class"""
    
    def __init__(self):
        self.volume_footprint = VolumeFootprintIndicator()
        self.instruments = {
            "NIFTY 50": {"id": 26000, "segment": "NSE_EQ", "type": "EQUITY"},
            "BANK NIFTY": {"id": 26001, "segment": "NSE_EQ", "type": "EQUITY"},
            "NIFTY INDEX": {"id": 13, "segment": "IDX_I", "type": "INDEX"}
        }
    
    def display_configuration(self):
        """Display configuration sidebar"""
        st.sidebar.markdown("## Configuration")
        
        # Instrument selection
        selected_instrument = st.sidebar.selectbox(
            "Select Instrument",
            list(self.instruments.keys()),
            index=2  # Default to NIFTY INDEX
        )
        
        # Update interval
        update_interval = st.sidebar.slider(
            "Update Interval (seconds)",
            min_value=5,
            max_value=60,
            value=30
        )
        
        # Data mode
        data_mode = st.sidebar.radio(
            "Select Data Mode",
            ["Historical Only", "Live Only", "Historical + Live"],
            index=0
        )
        
        # Historical data settings
        if data_mode in ["Historical Only", "Historical + Live"]:
            days_back = st.sidebar.slider(
                "Days of Historical Data",
                min_value=1,
                max_value=30,
                value=5
            )
            
            timeframe = st.sidebar.selectbox(
                "Timeframe",
                ["1", "5", "15", "25", "60", "Daily"],
                index=1  # Default to 5 minutes
            )
        else:
            days_back = 1
            timeframe = "5"
        
        # Volume footprint settings
        footprint_bins = st.sidebar.slider(
            "Volume Footprint Bins",
            min_value=10,
            max_value=50,
            value=20
        )
        
        return {
            "instrument": selected_instrument,
            "update_interval": update_interval,
            "data_mode": data_mode,
            "days_back": days_back,
            "timeframe": timeframe,
            "footprint_bins": footprint_bins
        }
    
    def display_status(self):
        """Display API status and metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Calls", st.session_state.api_call_count)
        
        with col2:
            last_update = st.session_state.last_update
            if last_update:
                time_diff = datetime.now(timezone('Asia/Kolkata')) - last_update
                st.metric("Last Update", f"{time_diff.seconds}s ago")
            else:
                st.metric("Last Update", "Never")
        
        with col3:
            hist_data_len = len(st.session_state.historical_data) if not st.session_state.historical_data.empty else 0
            st.metric("Historical Candles", hist_data_len)
        
        with col4:
            if dhan_config["client_id"]:
                st.metric("API Status", "Connected", delta="Active")
            else:
                st.metric("API Status", "Not Configured", delta="Error")
    
    def fetch_and_update_data(self, config):
        """Fetch and update historical and live data"""
        instrument_info = self.instruments[config["instrument"]]
        
        # Calculate date range
        end_date = datetime.now(timezone('Asia/Kolkata'))
        start_date = end_date - timedelta(days=config["days_back"])
        
        # Fetch historical data if needed
        if config["data_mode"] in ["Historical Only", "Historical + Live"]:
            historical_data = get_historical_data(
                instrument_info["id"],
                instrument_info["segment"],
                instrument_info["type"],
                start_date,
                end_date,
                config["timeframe"]
            )
            
            if not historical_data.empty:
                st.session_state.historical_data = historical_data
                st.session_state.last_update = datetime.now(timezone('Asia/Kolkata'))
        
        # Fetch live data if needed
        if config["data_mode"] in ["Live Only", "Historical + Live"]:
            live_data = get_live_data(
                instrument_info["id"],
                instrument_info["segment"]
            )
            
            if live_data:
                # Convert to DataFrame format
                live_df = pd.DataFrame([{
                    'datetime': live_data['timestamp'],
                    'open': live_data['open'],
                    'high': live_data['high'],
                    'low': live_data['low'],
                    'close': live_data['last_price'],
                    'volume': live_data['volume']
                }])
                
                st.session_state.live_data = live_df
    
    def update_volume_footprint_with_indicator(self, data_df):
        """Update volume footprint analysis with proper error handling"""
        if data_df.empty:
            st.warning("No data available for volume footprint analysis")
            return
        
        try:
            # Configure footprint bins
            self.volume_footprint.bins = 20  # Default bins
            
            # Update volume footprint
            session_high, session_low = self.volume_footprint.update_volume_footprint(data_df)
            
            # Store footprint data in session state
            if not self.volume_footprint.footprint_data.empty:
                st.session_state.footprint_data = self.volume_footprint.footprint_data
                
                # Display key levels
                if session_high and session_low:
                    st.info(f"Session High: {session_high:.2f} | Session Low: {session_low:.2f}")
            
        except Exception as e:
            st.error(f"Volume footprint analysis failed: {str(e)}")
    
    def display_charts(self):
        """Display trading charts"""
        # Combine historical and live data
        display_data = st.session_state.historical_data.copy()
        
        if not st.session_state.live_data.empty and not display_data.empty:
            # Append live data
            display_data = pd.concat([display_data, st.session_state.live_data], ignore_index=True)
            display_data = display_data.drop_duplicates(subset=['datetime']).sort_values('datetime')
        elif st.session_state.live_data.empty and display_data.empty:
            st.warning("No data available to display")
            return
        
        # Create charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not display_data.empty:
                candlestick_fig = create_candlestick_chart(display_data, "Price Chart with Volume Analysis")
                if candlestick_fig:
                    st.plotly_chart(candlestick_fig, use_container_width=True)
        
        with col2:
            if not st.session_state.footprint_data.empty:
                footprint_fig = create_volume_footprint_chart(st.session_state.footprint_data, "Volume Footprint")
                if footprint_fig:
                    st.plotly_chart(footprint_fig, use_container_width=True)
            else:
                st.info("Volume footprint data will appear here after analysis")
        
        # Display data table
        with st.expander("Raw Data"):
            if not display_data.empty:
                st.dataframe(display_data.tail(20))
            else:
                st.info("No data to display")

def main():
    """Main application function"""
    st.title("DhanHQ Trading Dashboard with Volume Footprint")
    
    # Check credentials
    if not dhan_config["client_id"] or not dhan_config["access_token"]:
        st.error("DhanHQ credentials not configured!")
        st.markdown("""
        ### Setup Instructions:
        1. Add your DhanHQ credentials to Streamlit secrets:
        ```
        DHAN_CLIENT_ID = "your_client_id"
        DHAN_ACCESS_TOKEN = "your_access_token"
        ```
        2. Get credentials from your DhanHQ account dashboard
        """)
        return
    
    # Current time display
    current_time = datetime.now(timezone('Asia/Kolkata'))
    st.markdown(f"**Current IST Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize dashboard
    dashboard = TradingDashboard()
    
    # Display configuration
    config = dashboard.display_configuration()
    
    # Display status
    dashboard.display_status()
    
    # Auto-refresh logic
    if st.sidebar.button("Fetch Data Now") or (
        config["data_mode"] in ["Live Only", "Historical + Live"] and 
        (st.session_state.last_update is None or 
         (current_time - st.session_state.last_update).seconds > config["update_interval"])
    ):
        # Fetch and update data
        dashboard.fetch_and_update_data(config)
        
        # Update volume footprint analysis
        if not st.session_state.historical_data.empty:
            dashboard.update_volume_footprint_with_indicator(st.session_state.historical_data)
    
    # Display charts
    dashboard.display_charts()
    
    # Export functionality
    st.sidebar.markdown("---")
    if st.sidebar.button("Export Data"):
        if not st.session_state.historical_data.empty:
            csv_data = st.session_state.historical_data.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"trading_data_{current_time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()