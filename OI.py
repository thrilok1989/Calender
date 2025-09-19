import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Streamlit page config
st.set_page_config(
    page_title="DhanHQ Candle Data",
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
            st.error(f"Missing secret: {e}. Please configure in Streamlit secrets.")
            st.stop()
        
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'Content-Type': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id
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
    
    def find_security_id(self, symbol, exchange_segment):
        """Find security ID for a symbol"""
        instruments = self.download_instrument_master()
        if instruments is None:
            return None
        
        # Map exchange segments
        segment_map = {
            'NSE_EQ': 'E',
            'NSE_FNO': 'D', 
            'BSE_EQ': 'E',
            'MCX_COMM': 'M'
        }
        
        segment_code = segment_map.get(exchange_segment, 'E')
        
        filtered = instruments[
            (instruments['SEM_CUSTOM_SYMBOL'].str.contains(symbol, case=False, na=False)) &
            (instruments['SEM_SEGMENT'] == segment_code)
        ]
        
        if not filtered.empty:
            return str(filtered.iloc[0]['SEM_EXM_EXCH_ID'])
        return None
    
    def get_historical_candles(self, security_id, exchange_segment, instrument, 
                             from_date, to_date, include_oi=False):
        """Get daily historical candles"""
        url = f"{self.base_url}/charts/historical"
        
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "expiryCode": 0,
            "oi": include_oi,
            "fromDate": from_date,
            "toDate": to_date
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return self._format_candle_data(data)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Request failed: {e}")
            return None
    
    def get_intraday_candles(self, security_id, exchange_segment, instrument,
                           interval, from_date, to_date, include_oi=False):
        """Get intraday candles"""
        url = f"{self.base_url}/charts/intraday"
        
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": include_oi,
            "fromDate": from_date,
            "toDate": to_date
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return self._format_candle_data(data)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Request failed: {e}")
            return None
    
    def _format_candle_data(self, raw_data):
        """Convert raw API response to DataFrame"""
        if not raw_data or 'open' not in raw_data:
            return None
        
        df = pd.DataFrame({
            'timestamp': raw_data['timestamp'],
            'open': raw_data['open'],
            'high': raw_data['high'],
            'low': raw_data['low'],
            'close': raw_data['close'],
            'volume': raw_data['volume']
        })
        
        if 'open_interest' in raw_data:
            df['open_interest'] = raw_data['open_interest']
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

def create_candlestick_chart(df, title):
    """Create candlestick chart with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
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
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def main():
    st.title("📈 DhanHQ Candle Data Dashboard")
    st.markdown("Real-time and historical candle data with volume analysis")
    
    # Initialize API
    dhan = DhanAPI()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Symbol input
    symbol = st.sidebar.text_input("Stock Symbol", value="RELIANCE").upper()
    
    # Exchange segment
    exchange_segment = st.sidebar.selectbox(
        "Exchange Segment",
        ["NSE_EQ", "NSE_FNO", "BSE_EQ", "MCX_COMM"]
    )
    
    # Instrument type
    instrument_map = {
        "NSE_EQ": "EQUITY",
        "NSE_FNO": ["FUTIDX", "OPTIDX", "FUTSTK", "OPTSTK"],
        "BSE_EQ": "EQUITY",
        "MCX_COMM": ["FUTCOM", "OPTFUT"]
    }
    
    if isinstance(instrument_map[exchange_segment], list):
        instrument = st.sidebar.selectbox("Instrument Type", instrument_map[exchange_segment])
    else:
        instrument = instrument_map[exchange_segment]
        st.sidebar.write(f"Instrument: {instrument}")
    
    # Data type selection
    data_type = st.sidebar.radio("Data Type", ["Historical Daily", "Intraday"])
    
    if data_type == "Intraday":
        interval = st.sidebar.selectbox("Interval", ["1", "5", "15", "25", "60"])
        st.sidebar.write("Note: Max 90 days for intraday data")
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        from_date = st.date_input("From Date", value=datetime.now() - timedelta(days=30))
    with col2:
        to_date = st.date_input("To Date", value=datetime.now())
    
    # Include OI for derivatives
    include_oi = st.sidebar.checkbox("Include Open Interest", value=False)
    
    # Fetch data button
    if st.sidebar.button("Fetch Data", type="primary"):
        with st.spinner("Fetching data..."):
            # Find security ID
            security_id = dhan.find_security_id(symbol, exchange_segment)
            
            if not security_id:
                st.error(f"Symbol '{symbol}' not found in {exchange_segment}")
                return
            
            st.success(f"Found {symbol} - Security ID: {security_id}")
            
            # Fetch candle data
            if data_type == "Historical Daily":
                df = dhan.get_historical_candles(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument=instrument,
                    from_date=from_date.strftime("%Y-%m-%d"),
                    to_date=to_date.strftime("%Y-%m-%d"),
                    include_oi=include_oi
                )
            else:
                # Format datetime for intraday
                from_datetime = f"{from_date.strftime('%Y-%m-%d')} 09:30:00"
                to_datetime = f"{to_date.strftime('%Y-%m-%d')} 15:30:00"
                
                df = dhan.get_intraday_candles(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument=instrument,
                    interval=interval,
                    from_date=from_datetime,
                    to_date=to_datetime,
                    include_oi=include_oi
                )
            
            if df is not None and not df.empty:
                # Store in session state
                st.session_state['df'] = df
                st.session_state['symbol'] = symbol
                st.session_state['data_type'] = data_type
                
                st.success(f"Fetched {len(df)} candles")
            else:
                st.error("No data received")
    
    # Display data if available
    if 'df' in st.session_state:
        df = st.session_state['df']
        symbol = st.session_state['symbol']
        data_type = st.session_state['data_type']
        
        # Chart
        chart_title = f"{symbol} - {data_type} Candles"
        fig = create_candlestick_chart(df, chart_title)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Candles", len(df))
        
        with col2:
            total_volume = df['volume'].sum()
            st.metric("Total Volume", f"{total_volume:,.0f}")
        
        with col3:
            avg_volume = df['volume'].mean()
            st.metric("Avg Volume", f"{avg_volume:,.0f}")
        
        with col4:
            price_change = df['close'].iloc[-1] - df['close'].iloc[0]
            st.metric("Price Change", f"{price_change:.2f}")
        
        # Data table
        st.subheader("Recent Data")
        display_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        if 'open_interest' in df.columns:
            display_cols.append('open_interest')
        
        st.dataframe(
            df[display_cols].tail(20).round(2),
            use_container_width=True
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{symbol}_{data_type}_candles.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()