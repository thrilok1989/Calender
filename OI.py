import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Streamlit page config
st.set_page_config(
    page_title="DhanHQ Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DhanAPI:
    def __init__(self):
        # Get credentials from Streamlit secrets
        try:
            self.access_token = st.secrets["DHAN_ACCESS_TOKEN"]
            self.client_id = st.secrets["DHAN_CLIENT_ID"]
        except KeyError as e:
            st.error(f"Missing secret: {e}. Please configure DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID in Streamlit secrets.")
            st.stop()
        
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'Content-Type': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id
        }
        
        # Predefined security IDs for popular instruments
        self.popular_instruments = {
            'NIFTY': {'id': '13', 'segment': 'IDX_I', 'instrument': 'INDEX'},
            'BANKNIFTY': {'id': '25', 'segment': 'IDX_I', 'instrument': 'INDEX'},
            'SENSEX': {'id': '51', 'segment': 'IDX_I', 'instrument': 'INDEX'},
            'RELIANCE': {'id': '1333', 'segment': 'NSE_EQ', 'instrument': 'EQUITY'},
            'TCS': {'id': '11536', 'segment': 'NSE_EQ', 'instrument': 'EQUITY'},
            'INFY': {'id': '1594', 'segment': 'NSE_EQ', 'instrument': 'EQUITY'},
            'HDFCBANK': {'id': '1333', 'segment': 'NSE_EQ', 'instrument': 'EQUITY'},
            'ICICIBANK': {'id': '4963', 'segment': 'NSE_EQ', 'instrument': 'EQUITY'}
        }
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def download_instrument_master(_self):
        """Download and cache instrument master"""
        try:
            url = "https://images.dhan.co/api-data/api-scrip-master.csv"
            df = pd.read_csv(url)
            return df
        except Exception as e:
            st.error(f"Error downloading instrument master: {e}")
            return None
    
    def get_instrument_info(self, symbol):
        """Get instrument info from predefined list or search"""
        symbol = symbol.upper()
        
        # Check predefined instruments first
        if symbol in self.popular_instruments:
            info = self.popular_instruments[symbol]
            return info['id'], info['segment'], info['instrument']
        
        # Search in instrument master
        instruments = self.download_instrument_master()
        if instruments is None:
            return None, None, None
        
        # Search by symbol
        filtered = instruments[
            instruments['SEM_CUSTOM_SYMBOL'].str.contains(symbol, case=False, na=False)
        ]
        
        if not filtered.empty:
            row = filtered.iloc[0]
            security_id = str(row['SEM_EXM_EXCH_ID'])
            
            # Determine exchange segment and instrument type
            segment = row['SEM_SEGMENT']
            segment_map = {
                'E': 'NSE_EQ',
                'D': 'NSE_FNO', 
                'C': 'NSE_CURRENCY',
                'M': 'MCX_COMM'
            }
            exchange_segment = segment_map.get(segment, 'NSE_EQ')
            
            # Determine instrument type
            instrument_name = str(row.get('SEM_INSTRUMENT_NAME', '')).upper()
            if 'INDEX' in instrument_name:
                instrument = 'INDEX'
                exchange_segment = 'IDX_I'
            elif 'EQUITY' in instrument_name:
                instrument = 'EQUITY'
            elif 'FUTURE' in instrument_name:
                instrument = 'FUTSTK' if exchange_segment == 'NSE_EQ' else 'FUTIDX'
            elif 'OPTION' in instrument_name:
                instrument = 'OPTSTK' if exchange_segment == 'NSE_EQ' else 'OPTIDX'
            else:
                instrument = 'EQUITY'  # Default
            
            return security_id, exchange_segment, instrument
        
        return None, None, None
    
    def validate_parameters(self, security_id, exchange_segment, instrument, from_date, to_date):
        """Validate API parameters"""
        if not security_id:
            st.error("Security ID is required")
            return False
        
        if not exchange_segment:
            st.error("Exchange segment is required")
            return False
        
        if not instrument:
            st.error("Instrument type is required")
            return False
        
        # Validate date range
        if from_date >= to_date:
            st.error("From date must be before to date")
            return False
        
        # Check if requesting too much intraday data
        date_diff = (to_date - from_date).days
        if date_diff > 90:
            st.warning("Intraday data is limited to 90 days. Adjusting date range.")
            return False
        
        return True
    
    def get_historical_candles(self, security_id, exchange_segment, instrument, 
                             from_date, to_date, include_oi=False):
        """Get daily historical candles"""
        if not self.validate_parameters(security_id, exchange_segment, instrument, from_date, to_date):
            return None
        
        url = f"{self.base_url}/charts/historical"
        
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "expiryCode": 0,
            "oi": include_oi and exchange_segment in ['NSE_FNO', 'BSE_FNO'],
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d")
        }
        
        # Debug information
        with st.expander("Debug - API Request"):
            st.json(payload)
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_candle_data(data)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Request timeout. Please try again.")
            return None
        except Exception as e:
            st.error(f"Request failed: {e}")
            return None
    
    def get_intraday_candles(self, security_id, exchange_segment, instrument,
                           interval, from_date, to_date, include_oi=False):
        """Get intraday candles"""
        if not self.validate_parameters(security_id, exchange_segment, instrument, from_date, to_date):
            return None
        
        url = f"{self.base_url}/charts/intraday"
        
        # Format datetime strings
        from_datetime = f"{from_date.strftime('%Y-%m-%d')} 09:30:00"
        to_datetime = f"{to_date.strftime('%Y-%m-%d')} 15:30:00"
        
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": include_oi and exchange_segment in ['NSE_FNO', 'BSE_FNO'],
            "fromDate": from_datetime,
            "toDate": to_datetime
        }
        
        # Debug information
        with st.expander("Debug - API Request"):
            st.json(payload)
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_candle_data(data)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Request timeout. Please try again.")
            return None
        except Exception as e:
            st.error(f"Request failed: {e}")
            return None
    
    def _format_candle_data(self, raw_data):
        """Convert raw API response to DataFrame"""
        if not raw_data or not isinstance(raw_data, dict):
            st.error("Invalid API response format")
            return None
        
        required_fields = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        if not all(field in raw_data for field in required_fields):
            st.error("Missing required fields in API response")
            return None
        
        # Check if arrays have data
        if not raw_data['open'] or len(raw_data['open']) == 0:
            st.error("No data returned from API")
            return None
        
        # Verify all arrays have same length
        arrays = [raw_data[field] for field in required_fields]
        lengths = [len(arr) for arr in arrays]
        
        if len(set(lengths)) > 1:
            st.error("Inconsistent data array lengths in API response")
            return None
        
        try:
            df = pd.DataFrame({
                'timestamp': raw_data['timestamp'],
                'open': raw_data['open'],
                'high': raw_data['high'],
                'low': raw_data['low'],
                'close': raw_data['close'],
                'volume': raw_data['volume']
            })
            
            # Add Open Interest if available
            if 'open_interest' in raw_data and raw_data['open_interest']:
                df['open_interest'] = raw_data['open_interest']
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            
            # Remove any rows with invalid timestamps
            df = df.dropna(subset=['datetime'])
            
            if df.empty:
                st.error("No valid data after processing")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return None

def create_candlestick_chart(df, title, symbol):
    """Create candlestick chart with volume"""
    if df is None or df.empty:
        st.error("No data available for charting")
        return None
    
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} - Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Volume chart
        colors = ['red' if close < open else 'green' 
                  for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df['datetime'],
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            yaxis_title='Price (₹)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=True,
            template='plotly_white'
        )
        
        # Format y-axis for better readability
        fig.update_yaxes(tickformat='.2f', row=1, col=1)
        fig.update_yaxes(tickformat='.0s', row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def main():
    st.title("📈 DhanHQ Trading Dashboard")
    st.markdown("Real-time and historical candle data with volume analysis")
    
    # Check market status
    now = datetime.now()
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=45, second=0, microsecond=0)
    
    if now.weekday() < 5 and market_open <= now <= market_close:
        st.success("🟢 Market is OPEN")
    else:
        st.info("🔴 Market is CLOSED")
    
    # Initialize API
    try:
        dhan = DhanAPI()
        st.sidebar.success("✅ DhanHQ API Connected")
    except Exception as e:
        st.error(f"Failed to initialize DhanHQ API: {e}")
        return
    
    # Sidebar configuration
    st.sidebar.header("📊 Configuration")
    
    # Symbol selection
    symbol_option = st.sidebar.selectbox(
        "Select Symbol",
        ["Custom"] + list(dhan.popular_instruments.keys())
    )
    
    if symbol_option == "Custom":
        symbol = st.sidebar.text_input("Enter Symbol", value="RELIANCE").upper()
    else:
        symbol = symbol_option
    
    # Data type selection
    data_type = st.sidebar.radio("Data Type", ["Historical Daily", "Intraday"])
    
    if data_type == "Intraday":
        interval = st.sidebar.selectbox("Interval (minutes)", ["1", "5", "15", "25", "60"])
        max_days = 30  # Limit for intraday
        st.sidebar.info("⚠️ Intraday data limited to 90 days")
    else:
        max_days = 365  # Limit for daily data
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        from_date = st.date_input(
            "From Date", 
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )
    with col2:
        to_date = st.date_input(
            "To Date", 
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Additional options
    include_oi = st.sidebar.checkbox("Include Open Interest", value=False)
    
    # Fetch data button
    if st.sidebar.button("📈 Fetch Data", type="primary"):
        with st.spinner(f"Fetching {data_type.lower()} data for {symbol}..."):
            
            # Get instrument information
            security_id, exchange_segment, instrument = dhan.get_instrument_info(symbol)
            
            if not security_id:
                st.error(f"❌ Symbol '{symbol}' not found. Please check the symbol name.")
                return
            
            # Display instrument info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Security ID:** {security_id}")
            with col2:
                st.info(f"**Exchange:** {exchange_segment}")
            with col3:
                st.info(f"**Instrument:** {instrument}")
            
            # Fetch candle data
            if data_type == "Historical Daily":
                df = dhan.get_historical_candles(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument=instrument,
                    from_date=from_date,
                    to_date=to_date,
                    include_oi=include_oi
                )
            else:
                df = dhan.get_intraday_candles(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument=instrument,
                    interval=interval,
                    from_date=from_date,
                    to_date=to_date,
                    include_oi=include_oi
                )
            
            if df is not None and not df.empty:
                # Store in session state
                st.session_state['df'] = df
                st.session_state['symbol'] = symbol
                st.session_state['data_type'] = data_type
                st.session_state['exchange_segment'] = exchange_segment
                st.session_state['instrument'] = instrument
                
                st.success(f"✅ Fetched {len(df)} candles for {symbol}")
            else:
                st.error("❌ No data received. Please check parameters and try again.")
    
    # Display data if available
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        symbol = st.session_state['symbol']
        data_type = st.session_state['data_type']
        
        # Statistics
        st.subheader("📊 Market Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Candles", len(df))
        
        with col2:
            total_volume = df['volume'].sum()
            st.metric("Total Volume", f"{total_volume:,.0f}")
        
        with col3:
            avg_volume = df['volume'].mean()
            st.metric("Avg Volume", f"{avg_volume:,.0f}")
        
        with col4:
            if len(df) > 1:
                price_change = df['close'].iloc[-1] - df['close'].iloc[0]
                price_change_pct = (price_change / df['close'].iloc[0]) * 100
                st.metric(
                    "Price Change", 
                    f"₹{price_change:.2f}",
                    f"{price_change_pct:.2f}%"
                )
            else:
                st.metric("Price Change", "N/A")
        
        with col5:
            current_price = df['close'].iloc[-1]
            st.metric("Current Price", f"₹{current_price:.2f}")
        
        # Chart
        st.subheader(f"📈 {symbol} - {data_type} Chart")
        chart_title = f"{symbol} - {data_type} Candles"
        fig = create_candlestick_chart(df, chart_title, symbol)
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Recent Data")
        display_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        if 'open_interest' in df.columns:
            display_cols.append('open_interest')
        
        # Format the dataframe for better display
        display_df = df[display_cols].tail(20).copy()
        display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Round numerical columns
        for col in ['open', 'high', 'low', 'close']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download section
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{symbol}_{data_type.replace(' ', '_')}_candles.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"{symbol}_{data_type.replace(' ', '_')}_candles.json",
                mime="application/json"
            )
    
    # Instructions
    with st.expander("📖 Instructions"):
        st.markdown("""
        **How to use:**
        1. Select a popular symbol or enter a custom one
        2. Choose between Historical Daily or Intraday data
        3. Set your date range (max 90 days for intraday)
        4. Click 'Fetch Data' to load charts
        5. Download data in CSV or JSON format
        
        **Popular Symbols:**
        - **Indices:** NIFTY, BANKNIFTY, SENSEX
        - **Stocks:** RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK
        
        **Data Includes:**
        - OHLC (Open, High, Low, Close) prices
        - Volume data for all instruments
        - Open Interest (for derivatives when selected)
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.write("Please refresh the page and try again.")