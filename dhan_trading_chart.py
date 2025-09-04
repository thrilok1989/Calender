import str eamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict
import warnings
from supabase import create_client, Client
import asyncio
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="DhanHQ Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SupabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
    def insert_market_data(self, data: Dict):
        """Insert market data into Supabase"""
        try:
            result = self.supabase.table('market_data').insert({
                'timestamp': data['timestamp'].isoformat(),
                'security_id': data.get('security_id', '11536'),
                'exchange_segment': data.get('exchange_segment', 'NSE_EQ'),
                'open': float(data['open']),
                'high': float(data['high']),
                'low': float(data['low']),
                'close': float(data['close']),
                'volume': int(data['volume']),
                'buy_quantity': int(data.get('buy_quantity', 0)),
                'sell_quantity': int(data.get('sell_quantity', 0))
            }).execute()
            return result.data
        except Exception as e:
            st.error(f"Database error: {e}")
            return None
    
    def insert_volume_footprint(self, security_id: str, footprint_data: Dict, session_high: float, session_low: float):
        """Insert volume footprint data"""
        try:
            records = []
            poc_price = max(footprint_data, key=footprint_data.get) if footprint_data else None
            
            for price_level, volume in footprint_data.items():
                records.append({
                    'timestamp': datetime.now().isoformat(),
                    'security_id': security_id,
                    'exchange_segment': 'NSE_EQ',
                    'price_level': float(price_level),
                    'volume': float(volume),
                    'is_poc': price_level == poc_price,
                    'session_high': float(session_high),
                    'session_low': float(session_low)
                })
            
            if records:
                result = self.supabase.table('volume_footprint').insert(records).execute()
                return result.data
        except Exception as e:
            st.error(f"Volume footprint database error: {e}")
            return None
    
    def insert_alert(self, alert_type: str, message: str, security_id: str, price: float = None, volume: int = None):
        """Insert alert into database"""
        try:
            result = self.supabase.table('alerts').insert({
                'timestamp': datetime.now().isoformat(),
                'alert_type': alert_type,
                'message': message,
                'security_id': security_id,
                'exchange_segment': 'NSE_EQ',
                'price': float(price) if price else None,
                'volume': int(volume) if volume else None,
                'is_sent': False
            }).execute()
            return result.data
        except Exception as e:
            st.error(f"Alert database error: {e}")
            return None
    
    def get_recent_market_data(self, security_id: str, limit: int = 100):
        """Get recent market data from database"""
        try:
            result = self.supabase.table('market_data')\
                .select('*')\
                .eq('security_id', security_id)\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            return result.data
        except Exception as e:
            st.error(f"Error fetching market data: {e}")
            return []

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str, parse_mode: str = "HTML"):
        """Send message via Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Telegram error: {response.text}")
                return None
        except Exception as e:
            st.error(f"Telegram exception: {e}")
            return None
    
    def send_chart_alert(self, alert_type: str, security_id: str, price: float, volume: int, additional_info: str = ""):
        """Send formatted trading alert"""
        message = f"""
🚨 <b>{alert_type.upper()} ALERT</b> 🚨

📊 <b>Security:</b> {security_id}
💰 <b>Price:</b> ₹{price:.2f}
📈 <b>Volume:</b> {volume:,}
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

{additional_info}

#TradingAlert #{security_id}
        """.strip()
        
        return self.send_message(message)

class DhanTradingDashboard:
    def __init__(self):
        # Initialize from Streamlit secrets
        self.access_token = st.secrets["DHAN_ACCESS_TOKEN"]
        self.client_id = st.secrets["DHAN_CLIENT_ID"]
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id
        }
        
        # Initialize database and notifications
        self.db = SupabaseManager(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        self.telegram = TelegramNotifier(st.secrets["TELEGRAM_BOT_TOKEN"], st.secrets["TELEGRAM_CHAT_ID"])
        
        # Volume Footprint settings
        self.bins = 20
        self.timeframe = '1D'
        
        # Alert thresholds
        self.volume_spike_threshold = 2.0
        self.price_change_threshold = 0.02
        
        # Session data
        if 'current_high' not in st.session_state:
            st.session_state.current_high = None
            st.session_state.current_low = None
            st.session_state.volume_profile = defaultdict(float)
            st.session_state.chart_data = pd.DataFrame()
            st.session_state.previous_poc = None
            st.session_state.last_update = None
    
    def fetch_historical_data(self, security_id="13", exchange_segment="IDX_I", 
                            instrument="INDEX", from_date=None, to_date=None, 
                            interval=None, oi=False):
        """
        Fetch historical data from DhanHQ API
        
        Parameters:
        - security_id: Exchange standard ID for each scrip
        - exchange_segment: Exchange & segment for which data is to be fetched
        - instrument: Instrument type of the scrip
        - from_date: Start date of the desired range (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        - to_date: End date of the desired range (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        - interval: For intraday data - 1, 5, 15, 25, 60 minutes
        - oi: Boolean to include Open Interest data for Futures & Options
        """
        try:
            if interval:
                # Intraday historical data
                url = f"{self.base_url}/charts/intraday"
                payload = {
                    "securityId": security_id,
                    "exchangeSegment": exchange_segment,
                    "instrument": instrument,
                    "interval": str(interval),
                    "oi": oi,
                    "fromDate": from_date,
                    "toDate": to_date
                }
            else:
                # Daily historical data
                url = f"{self.base_url}/charts/historical"
                payload = {
                    "securityId": security_id,
                    "exchangeSegment": exchange_segment,
                    "instrument": instrument,
                    "expiryCode": 0,
                    "oi": oi,
                    "fromDate": from_date,
                    "toDate": to_date
                }
            
            st.info(f"Fetching historical data for {security_id} from {exchange_segment}...")
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if not data or 'timestamp' not in data or not data['timestamp']:
                    st.warning("No historical data available for this instrument")
                    return self.create_sample_data(security_id, exchange_segment, instrument)
                
                # Convert to DataFrame
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume']
                })
                
                # Add buy/sell quantities as placeholders
                df['buy_quantity'] = 0
                df['sell_quantity'] = 0
                
                # Add open interest if available
                if 'open_interest' in data and any(data['open_interest']):
                    df['open_interest'] = data['open_interest']
                
                st.success(f"Loaded {len(df)} historical data points")
                return df
            elif response.status_code == 429:
                st.warning("Rate limit reached. Using sample data.")
                return self.create_sample_data(security_id, exchange_segment, instrument)
            else:
                st.error(f"Historical Data API Error: {response.status_code} - {response.text}")
                return self.create_sample_data(security_id, exchange_segment, instrument)
                
        except Exception as e:
            st.error(f"Exception in Historical Data API call: {e}")
            return self.create_sample_data(security_id, exchange_segment, instrument)
    
    def create_sample_data(self, security_id, exchange_segment, instrument):
        """Create sample data when API is unavailable"""
        try:
            # Base prices for different instruments
            base_prices = {
                "13": 24000,    # Nifty 50
                "51": 80000,    # Sensex
                "25": 52000,    # Bank Nifty
                "27": 35000,    # Nifty IT
                "1333": 3000,   # Reliance
                "3431": 800,    # Tata Motors
            }
            
            base_price = base_prices.get(security_id, 4500)
            
            # Generate 30 days of sample data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            # Create timestamps
            timestamps = []
            current_time = start_time
            
            while current_time < end_time:
                if current_time.weekday() < 5:  # Weekdays only
                    timestamps.append(current_time)
                current_time += timedelta(hours=1)
            
            # Generate OHLC data
            np.random.seed(42)
            data = []
            current_price = base_price
            
            for i, timestamp in enumerate(timestamps):
                volatility = base_price * 0.01
                price_change = np.random.normal(0, volatility)
                
                open_price = current_price
                close_price = open_price + price_change
                
                high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility/2))
                low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility/2))
                
                volume = int(np.random.normal(100000, 30000))
                volume = max(volume, 1000)
                
                data.append({
                    'timestamp': timestamp,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'buy_quantity': 0,
                    'sell_quantity': 0
                })
                
                current_price = close_price
            
            df = pd.DataFrame(data)
            st.info(f"Using sample data with {len(df)} candles for {security_id}")
            
            return df
            
        except Exception as e:
            st.error(f"Error creating sample data: {e}")
            return pd.DataFrame()
    
    def fetch_market_data(self, security_id="13", exchange_segment="IDX_I"):
        """Fetch real-time market data using DhanHQ API"""
        try:
            quote_url = f"{self.base_url}/marketfeed/quote"
            quote_payload = {exchange_segment: [int(security_id)]}
            
            response = requests.post(quote_url, headers=self.headers, json=quote_payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and exchange_segment in data['data']:
                    market_data = data['data'][exchange_segment][security_id]
                    processed_data = self.process_market_data(market_data)
                    
                    if processed_data:
                        processed_data['security_id'] = security_id
                        processed_data['exchange_segment'] = exchange_segment
                        
                        # Store in database
                        try:
                            self.db.insert_market_data(processed_data)
                        except:
                            pass  # Continue even if database fails
                    
                    return processed_data
            elif response.status_code == 429:
                st.warning("API rate limit reached. Continuing with historical data...")
                return None
            else:
                st.warning(f"API temporarily unavailable (Status: {response.status_code})")
                return None
                
        except Exception as e:
            st.warning(f"Real-time API unavailable: {str(e)[:100]}...")
            return None
    
    def process_market_data(self, raw_data):
        """Process raw market data into structured format"""
        try:
            current_time = datetime.now()
            
            processed_data = {
                'timestamp': current_time,
                'open': raw_data['ohlc']['open'],
                'high': raw_data['ohlc']['high'],
                'low': raw_data['ohlc']['low'],
                'close': raw_data['last_price'],
                'volume': raw_data['volume'],
                'buy_quantity': raw_data.get('buy_quantity', 0),
                'sell_quantity': raw_data.get('sell_quantity', 0)
            }
            
            return processed_data
            
        except Exception as e:
            st.error(f"Error processing market data: {e}")
            return None
    
    def update_volume_footprint(self, market_data):
        """Update volume footprint data"""
        if not market_data:
            return
            
        current_price = market_data['close']
        volume = market_data['volume']
        high = market_data['high']
        low = market_data['low']
        
        # Initialize session if needed
        if st.session_state.current_high is None or st.session_state.current_low is None:
            st.session_state.current_high = high
            st.session_state.current_low = low
        else:
            # Update session high/low
            if high > st.session_state.current_high:
                st.session_state.current_high = high
            if low < st.session_state.current_low:
                st.session_state.current_low = low
        
        # Calculate volume footprint
        if st.session_state.current_high > st.session_state.current_low:
            price_range = st.session_state.current_high - st.session_state.current_low
            step = price_range / self.bins if self.bins > 0 else 1
            
            # Volume weighted calculation
            recent_data = st.session_state.chart_data.tail(200) if len(st.session_state.chart_data) > 0 else pd.DataFrame()
            if len(recent_data) > 1:
                volume_stdev = recent_data['volume'].std()
                volume_val = volume / volume_stdev if volume_stdev > 0 else volume
            else:
                volume_val = volume
            
            # Find which bin this price falls into
            if step > 0:
                bin_index = int((current_price - st.session_state.current_low) / step)
                bin_index = max(0, min(bin_index, self.bins - 1))
                
                # Update volume profile
                bin_price = st.session_state.current_low + (bin_index * step)
                st.session_state.volume_profile[bin_price] += volume_val
        
        # Store volume footprint in database
        if market_data.get('security_id'):
            self.db.insert_volume_footprint(
                market_data['security_id'],
                dict(st.session_state.volume_profile),
                st.session_state.current_high,
                st.session_state.current_low
            )
    
    def calculate_poc(self):
        """Calculate Point of Control"""
        if not st.session_state.volume_profile:
            return None
            
        poc_price = max(st.session_state.volume_profile, key=st.session_state.volume_profile.get)
        poc_volume = st.session_state.volume_profile[poc_price]
        
        return {'price': poc_price, 'volume': poc_volume}
    
    def check_alerts(self, market_data, security_id):
        """Check for trading alerts"""
        if not market_data or len(st.session_state.chart_data) < 10:
            return
        
        current_price = market_data['close']
        current_volume = market_data['volume']
        
        # Calculate average volume
        recent_volumes = st.session_state.chart_data.tail(20)['volume']
        avg_volume = recent_volumes.mean() if len(recent_volumes) > 0 else current_volume
        
        # Volume spike alert
        if current_volume > avg_volume * self.volume_spike_threshold and avg_volume > 0:
            alert_msg = f"Volume spike: {current_volume:,} vs avg {avg_volume:,.0f}"
            self.db.insert_alert("VOLUME_SPIKE", alert_msg, security_id, current_price, current_volume)
            self.telegram.send_chart_alert("VOLUME_SPIKE", security_id, current_price, current_volume, alert_msg)
            st.success(f"🚨 Volume Spike Alert: {alert_msg}")
        
        # Price change alert
        if len(st.session_state.chart_data) > 1:
            previous_price = st.session_state.chart_data.iloc[-1]['close']
            price_change_pct = abs((current_price - previous_price) / previous_price)
            
            if price_change_pct > self.price_change_threshold:
                direction = "UP" if current_price > previous_price else "DOWN"
                alert_msg = f"Price movement {direction}: {price_change_pct*100:.1f}%"
                self.db.insert_alert("PRICE_CHANGE", alert_msg, security_id, current_price, current_volume)
                self.telegram.send_chart_alert("PRICE_CHANGE", security_id, current_price, current_volume, alert_msg)
                st.warning(f"📈 Price Change Alert: {alert_msg}")
    
    def create_candlestick_chart(self, data):
        """Create candlestick chart with volume footprint using Plotly"""
        if data is None or len(data) == 0:
            return None
            
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price Action with Volume Footprint', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=data['timestamp'],
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add POC line if available
        poc_data = self.calculate_poc()
        if poc_data:
            fig.add_hline(
                y=poc_data['price'],
                line_dash="dash",
                line_color="#298ada",
                annotation_text=f"POC: {poc_data['price']:.2f}",
                row=1, col=1
            )
        
        # Add session high/low lines
        if st.session_state.current_high and st.session_state.current_low:
            fig.add_hline(
                y=st.session_state.current_high,
                line_color="green",
                line_width=1,
                annotation_text=f"High: {st.session_state.current_high:.2f}",
                row=1, col=1
            )
            fig.add_hline(
                y=st.session_state.current_low,
                line_color="red", 
                line_width=1,
                annotation_text=f"Low: {st.session_state.current_low:.2f}",
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="DhanHQ Real-time Trading Dashboard",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template="plotly_dark"
        )
        
        return fig
    
    def create_volume_footprint_chart(self):
        """Create volume footprint visualization"""
        if not st.session_state.volume_profile:
            return None
        
        prices = list(st.session_state.volume_profile.keys())
        volumes = list(st.session_state.volume_profile.values())
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=volumes,
            y=prices,
            orientation='h',
            marker_color='#298ada',
            name='Volume Profile'
        ))
        
        # Highlight POC
        poc_data = self.calculate_poc()
        if poc_data:
            max_volume_idx = volumes.index(max(volumes))
            fig.data[0].marker.color = ['red' if i == max_volume_idx else '#298ada' 
                                       for i in range(len(volumes))]
        
        fig.update_layout(
            title="Volume Footprint",
            xaxis_title="Volume",
            yaxis_title="Price Level",
            template="plotly_dark",
            height=400
        )
        
        return fig
    
    def update_data(self, security_id, exchange_segment):
        """Update market data with rate limit handling"""
        # Check if we should attempt real-time fetch
        last_api_call = st.session_state.get(f'last_api_call_{security_id}', datetime.min)
        time_since_last_call = (datetime.now() - last_api_call).total_seconds()
        
        market_data = None
        
        # Only call API if enough time has passed (25+ seconds)
        if time_since_last_call >= 25:
            market_data = self.fetch_market_data(security_id, exchange_segment)
            st.session_state[f'last_api_call_{security_id}'] = datetime.now()
            
            if market_data:
                # Add to session state chart data
                new_row = pd.DataFrame([market_data])
                if len(st.session_state.chart_data) == 0:
                    st.session_state.chart_data = new_row
                else:
                    st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_row], ignore_index=True)
                
                # Keep only recent data (last 1000 points)
                if len(st.session_state.chart_data) > 1000:
                    st.session_state.chart_data = st.session_state.chart_data.tail(1000).reset_index(drop=True)
                
                # Update volume footprint
                self.update_volume_footprint(market_data)
                
                # Check for alerts
                self.check_alerts(market_data, security_id)
                
                st.session_state.last_update = datetime.now()
        
        # Return market data or latest historical data
        if market_data:
            return market_data
        elif len(st.session_state.chart_data) > 0:
            # Return the latest data point
            latest = st.session_state.chart_data.iloc[-1].to_dict()
            return latest
        
        return None

def main():
    st.title("📈 DhanHQ Real-time Trading Dashboard")
    
    # Check if secrets are configured
    try:
        dashboard = DhanTradingDashboard()
    except KeyError as e:
        st.error(f"Missing secret: {e}")
        st.info("Please configure the following secrets in your Streamlit app:")
        st.code("""
DHAN_ACCESS_TOKEN = "your_dhan_access_token"
DHAN_CLIENT_ID = "your_dhan_client_id"
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_telegram_chat_id"
        """)
        return
    
    # Correct instrument mapping with proper security IDs
    instruments = {
        "NIFTY 50": {"security_id": "13", "exchange_segment": "IDX_I", "instrument": "INDEX"},
        "SENSEX": {"security_id": "51", "exchange_segment": "IDX_I", "instrument": "INDEX"},
        "BANK NIFTY": {"security_id": "25", "exchange_segment": "IDX_I", "instrument": "INDEX"},
        "NIFTY IT": {"security_id": "27", "exchange_segment": "IDX_I", "instrument": "INDEX"},
        "RELIANCE": {"security_id": "1333", "exchange_segment": "NSE_EQ", "instrument": "EQUITY"},
        "TATA MOTORS": {"security_id": "3431", "exchange_segment": "NSE_EQ", "instrument": "EQUITY"},
    }
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Let user select from available instruments or enter custom ID
    instrument_options = list(instruments.keys()) + ["Custom"]
    selected_instrument = st.sidebar.selectbox("Select Instrument", instrument_options)
    
    if selected_instrument == "Custom":
        security_id = st.sidebar.text_input("Security ID", value="13")
        exchange_segment = st.sidebar.selectbox(
            "Exchange Segment",
            ["IDX_I", "NSE_EQ", "BSE_EQ", "NSE_FNO", "BSE_FNO", "MCX_FO"]
        )
        instrument_type = st.sidebar.selectbox(
            "Instrument Type",
            ["INDEX", "EQUITY", "FUTURES", "OPTIONS", "CURRENCY"]
        )
        selected_instrument_name = f"Custom ({security_id})"
    else:
        security_id = instruments[selected_instrument]["security_id"]
        exchange_segment = instruments[selected_instrument]["exchange_segment"]
        instrument_type = instruments[selected_instrument]["instrument"]
        selected_instrument_name = selected_instrument
    
    update_interval = st.sidebar.slider("Update Interval (seconds)", 20, 60, 23)
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Historical data section
    st.sidebar.header("Historical Data")
    historical_days = st.sidebar.slider("Days of Historical Data", 1, 90, 30)
    use_intraday = st.sidebar.checkbox("Use Intraday Data (5min)", value=True)
    
    if st.sidebar.button("Load Historical Data"):
        with st.spinner("Loading historical data..."):
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=historical_days)).strftime("%Y-%m-%d")
            
            if use_intraday:
                from_date += " 09:15:00"
                to_date += " 15:30:00"
                interval = "5"
            else:
                interval = None
            
            historical_data = dashboard.fetch_historical_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument=instrument_type,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if historical_data is not None and len(historical_data) > 0:
                st.session_state.chart_data = historical_data
                
                # Initialize volume footprint from historical data
                if len(historical_data) > 0:
                    st.session_state.current_high = historical_data['high'].max()
                    st.session_state.current_low = historical_data['low'].min()
                    
                    # Calculate volume footprint from historical data
                    st.session_state.volume_profile = defaultdict(float)
                    price_range = st.session_state.current_high - st.session_state.current_low
                    
                    if price_range > 0:
                        step = price_range / dashboard.bins
                        for _, row in historical_data.iterrows():
                            if step > 0:
                                bin_index = int((row['close'] - st.session_state.current_low) / step)
                                bin_index = max(0, min(bin_index, dashboard.bins - 1))
                                bin_price = st.session_state.current_low + (bin_index * step)
                                st.session_state.volume_profile[bin_price] += row['volume']
                
                st.success(f"Loaded {len(historical_data)} historical data points for {selected_instrument_name}")
            else:
                st.error("Failed to load historical data")
    
    # Auto-load historical data on first run
    if 'data_initialized' not in st.session_state:
        st.session_state.data_initialized = True
        with st.spinner(f"Auto-loading historical data for {selected_instrument_name}..."):
            to_date = datetime.now().strftime("%Y-%m-%d 15:30:00")
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d 09:15:00")
            
            historical_data = dashboard.fetch_historical_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument=instrument_type,
                from_date=from_date,
                to_date=to_date,
                interval="5"
            )
            
            if historical_data is not None and len(historical_data) > 0:
                st.session_state.chart_data = historical_data
                
                # Initialize volume footprint from historical data
                if len(historical_data) > 0:
                    st.session_state.current_high = historical_data['high'].max()
                    st.session_state.current_low = historical_data['low'].min()
                    
                    # Calculate volume footprint from historical data
                    st.session_state.volume_profile = defaultdict(float)
                    price_range = st.session_state.current_high - st.session_state.current_low
                    
                    if price_range > 0:
                        step = price_range / dashboard.bins
                        for _, row in historical_data.iterrows():
                            if step > 0:
                                bin_index = int((row['close'] - st.session_state.current_low) / step)
                                bin_index = max(0, min(bin_index, dashboard.bins - 1))
                                bin_price = st.session_state.current_low + (bin_index * step)
                                st.session_state.volume_profile[bin_price] += row['volume']
    
    # Multi-instrument tracking option
    track_multiple = st.sidebar.checkbox("Track Multiple Instruments", value=False)
    
    if track_multiple:
        st.sidebar.subheader("Additional Instruments")
        additional_instruments = st.sidebar.multiselect(
            "Select additional instruments to track",
            [key for key in instruments.keys() if key != selected_instrument],
            default=[]
        )
    
    # Main update logic
    if st.sidebar.button("Manual Update") or auto_refresh:
        # Update primary instrument
        market_data = dashboard.update_data(security_id, exchange_segment)
        
        # Store additional instruments data
        additional_data = {}
        if track_multiple and additional_instruments:
            for instrument_name in additional_instruments:
                inst_config = instruments[instrument_name]
                additional_data[instrument_name] = dashboard.update_data(
                    inst_config["security_id"], 
                    inst_config["exchange_segment"]
                )
        
        if market_data or len(st.session_state.chart_data) > 0:
            # Main layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if len(st.session_state.chart_data) > 0:
                    try:
                        chart = dashboard.create_candlestick_chart(st.session_state.chart_data)
                        if chart:
                            chart.update_layout(title=f"{selected_instrument_name} - Real-time Chart")
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.error("Unable to create chart")
                    except Exception as e:
                        st.error(f"Chart error: {e}")
                else:
                    st.info("Loading chart data...")
            
            with col2:
                st.subheader("Market Info")
                if market_data:
                    st.metric("Current Price", f"₹{market_data['close']:.2f}")
                    st.metric("Volume", f"{market_data['volume']:,}")
                    st.metric("Buy Qty", f"{market_data.get('buy_quantity', 0):,}")
                    st.metric("Sell Qty", f"{market_data.get('sell_quantity', 0):,}")
                elif len(st.session_state.chart_data) > 0:
                    latest = st.session_state.chart_data.iloc[-1]
                    st.metric("Last Price", f"₹{latest['close']:.2f}")
                    st.metric("Volume", f"{latest['volume']:,}")
                
                if st.session_state.current_high and st.session_state.current_low:
                    st.metric("Session High", f"₹{st.session_state.current_high:.2f}")
                    st.metric("Session Low", f"₹{st.session_state.current_low:.2f}")
                
                poc_data = dashboard.calculate_poc()
                if poc_data:
                    st.metric("POC Price", f"₹{poc_data['price']:.2f}")
                    st.metric("POC Volume", f"{poc_data['volume']:.1f}")
            
            # Additional instruments display
            if track_multiple and additional_data and any(additional_data.values()):
                st.subheader("📊 Additional Instruments")
                cols = st.columns(len(additional_instruments))
                
                for idx, (instrument_name, inst_data) in enumerate(additional_data.items()):
                    with cols[idx]:
                        st.subheader(instrument_name)
                        if inst_data:
                            st.metric("Price", f"₹{inst_data['close']:.2f}")
                            st.metric("Volume", f"{inst_data['volume']:,}")
            
            # Volume footprint chart
            st.subheader(f"Volume Footprint - {selected_instrument_name}")
            footprint_chart = dashboard.create_volume_footprint_chart()
            if footprint_chart:
                st.plotly_chart(footprint_chart, use_container_width=True)
            else:
                st.info("Volume footprint will appear after accumulating more data.")
            
            # Recent data table
            st.subheader(f"Recent Data - {selected_instrument_name}")
            if len(st.session_state.chart_data) > 0:
                recent_data = st.session_state.chart_data.tail(10).copy()
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%H:%M:%S')
                recent_data = recent_data.round(2)
                st.dataframe(recent_data, use_container_width=True)
            
            # Status info
            st.sidebar.success(f"✅ Connected to DhanHQ API")
            st.sidebar.info(f"Instrument: {selected_instrument_name}")
            st.sidebar.info(f"Security ID: {security_id}")
            st.sidebar.info(f"Exchange: {exchange_segment}")
            
            if st.session_state.last_update:
                st.sidebar.info(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
            
            # Data status
            if len(st.session_state.chart_data) > 0:
                st.sidebar.metric("Data Points", len(st.session_state.chart_data))
                data_start = st.session_state.chart_data['timestamp'].min()
                data_end = st.session_state.chart_data['timestamp'].max()
                st.sidebar.info(f"Data: {data_start.strftime('%m/%d %H:%M')} to {data_end.strftime('%m/%d %H:%M')}")
            
            # Price range info
            if len(st.session_state.chart_data) > 0:
                price_range = st.session_state.chart_data['close'].max() - st.session_state.chart_data['close'].min()
                st.sidebar.metric("Price Range", f"₹{price_range:.2f}")
            
            # Auto-refresh functionality
            if auto_refresh:
                time.sleep(update_interval)
                st.rerun()
        else:
            st.error("Failed to fetch market data. Please check your API credentials.")
            st.info("Try loading historical data manually from the sidebar.")

if __name__ == "__main__":
    main()
