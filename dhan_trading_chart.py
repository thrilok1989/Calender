import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import time
import threading
from collections import defaultdict
import warnings
from supabase import create_client, Client
import asyncio
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Page config
st.set_page_config(
    page_title="DhanHQ Trading Dashboard - IST",
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
                    'timestamp': datetime.now(IST).isoformat(),
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
        current_time = datetime.now(IST).strftime('%H:%M:%S IST')
        message = f"""
🚨 <b>{alert_type.upper()} ALERT</b> 🚨

📊 <b>Security:</b> {security_id}
💰 <b>Price:</b> ₹{price:.2f}
📈 <b>Volume:</b> {volume:,}
⏰ <b>Time:</b> {current_time}

{additional_info}

#TradingAlert #{security_id}
        """.strip()
        
        return self.send_message(message)

class VolumeFootprintIndicator:
    """Pine Script Volume Footprint Indicator converted to Python"""
    
    def __init__(self, bins=20, timeframe='1D'):
        self.bins = bins
        self.timeframe = timeframe
        self.poc_color = '#298ada'
        self.bg_color = 'rgba(120, 123, 134, 0.9)'
        
        # State variables (equivalent to Pine Script var declarations)
        self.source_h = None
        self.source_l = None
        self.src_h = []
        self.src_l = []
        self.array_freq = [0.0] * bins
        self.volume_profile = defaultdict(float)
        self.current_session_start = None
        
    def detect_htf_change(self, current_data, previous_data):
        """Detect higher timeframe change"""
        if not previous_data or not current_data:
            return True
            
        current_time = pd.to_datetime(current_data['timestamp'])
        previous_time = pd.to_datetime(previous_data['timestamp'])
        
        if self.timeframe == '1D':
            return current_time.date() != previous_time.date()
        elif self.timeframe == '1W':
            return current_time.isocalendar()[1] != previous_time.isocalendar()[1]
        elif self.timeframe == '1M':
            return current_time.month != previous_time.month
        else:
            return False
    
    def update_volume_footprint(self, data_df):
        """Update volume footprint based on Pine Script logic"""
        if data_df.empty:
            return None, None
            
        current_data = data_df.iloc[-1]
        previous_data = data_df.iloc[-2] if len(data_df) > 1 else None
        
        htf_change = self.detect_htf_change(current_data, previous_data)
        
        if htf_change:
            # Reset for new session
            self.src_h = []
            self.src_l = []
            self.array_freq = [0.0] * self.bins
            self.volume_profile = defaultdict(float)
            self.current_session_start = current_data['timestamp']
        
        # Update high and low arrays
        self.src_h.append(current_data['high'])
        self.src_l.append(current_data['low'])
        
        # Calculate session high and low
        self.source_h = max(self.src_h) if self.src_h else current_data['high']
        self.source_l = min(self.src_l) if self.src_l else current_data['low']
        
        # Update high and lows as in Pine Script
        if current_data['high'] > self.source_h:
            self.source_h = current_data['high']
        if current_data['low'] < self.source_l:
            self.source_l = current_data['low']
        
        # Calculate step size
        price_range = self.source_h - self.source_l
        step = price_range / self.bins if price_range > 0 else 0
        
        if step > 0:
            # Calculate normalized volume (equivalent to Pine Script volume / ta.stdev(volume, 200))
            volume_series = data_df['volume'].tail(200)
            volume_stdev = volume_series.std() if len(volume_series) > 1 else 1
            volume_val = current_data['volume'] / volume_stdev if volume_stdev > 0 else current_data['volume']
            
            # Update frequency array
            for i in range(self.bins):
                lower = self.source_l + i * step
                upper = lower + step
                
                # Check if current close price falls in this bin
                if current_data['close'] >= lower and current_data['close'] < upper:
                    self.array_freq[i] += volume_val
                    self.volume_profile[lower] = self.array_freq[i]
        
        return self.source_h, self.source_l
    
    def get_poc(self):
        """Get Point of Control"""
        if not self.volume_profile:
            return None
            
        poc_price = max(self.volume_profile, key=self.volume_profile.get)
        poc_volume = self.volume_profile[poc_price]
        
        return {'price': poc_price, 'volume': poc_volume}
    
    def get_volume_profile_data(self):
        """Get volume profile data for visualization"""
        if not self.volume_profile:
            return [], []
            
        prices = list(self.volume_profile.keys())
        volumes = list(self.volume_profile.values())
        
        return prices, volumes

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
        
        # Initialize Volume Footprint Indicator
        self.volume_footprint_indicator = VolumeFootprintIndicator(bins=20, timeframe='1D')
        
        # Alert thresholds
        self.volume_spike_threshold = 2.0
        self.price_change_threshold = 0.02
        
        # Session data initialization
        if 'chart_data' not in st.session_state:
            st.session_state.chart_data = pd.DataFrame()
            st.session_state.current_high = None
            st.session_state.current_low = None
            st.session_state.last_update = None
    
    def get_ist_time(self):
        """Get current IST time"""
        return datetime.now(IST)
    
    def fetch_historical_data(self, security_id="13", exchange_segment="IDX_I", 
                            instrument="INDEX", from_date=None, to_date=None, 
                            interval=None, oi=False):
        """Fetch historical data from DhanHQ API with IST timezone"""
        try:
            if interval:
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
                
                # Convert to DataFrame with IST timezone
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['timestamp'], unit='s', utc=True).dt.tz_convert(IST),
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
        """Create sample data with IST timezone"""
        try:
            base_prices = {
                "13": 24000,    # Nifty 50
                "51": 80000,    # Sensex
                "25": 52000,    # Bank Nifty
                "27": 35000,    # Nifty IT
                "1333": 3000,   # Reliance
                "3431": 800,    # Tata Motors
            }
            
            base_price = base_prices.get(security_id, 4500)
            
            # Generate sample data with IST timezone
            end_time = self.get_ist_time()
            start_time = end_time - timedelta(days=30)
            
            timestamps = []
            current_time = start_time
            
            while current_time < end_time:
                if current_time.weekday() < 5:  # Weekdays only
                    # Market hours: 9:15 AM to 3:30 PM IST
                    if 9 <= current_time.hour <= 15:
                        timestamps.append(current_time)
                current_time += timedelta(minutes=5)
            
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
            st.info(f"Using sample data with {len(df)} candles for {security_id} (IST timezone)")
            
            return df
            
        except Exception as e:
            st.error(f"Error creating sample data: {e}")
            return pd.DataFrame()
    
    def fetch_market_data(self, security_id="13", exchange_segment="IDX_I"):
        """Fetch real-time market data with IST timestamp"""
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
        """Process raw market data with IST timestamp"""
        try:
            current_time = self.get_ist_time()
            
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
    
    def update_volume_footprint_with_indicator(self, data_df):
        """Update volume footprint using Pine Script indicator logic"""
        if data_df.empty:
            return
            
        session_high, session_low = self.volume_footprint_indicator.update_volume_footprint(data_df)
        
        if session_high and session_low:
            st.session_state.current_high = session_high
            st.session_state.current_low = session_low
    
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
            self.telegram.send_chart_alert("VOLUME_SPIKE", security_id, current_price, current_volume, alert_msg)
            st.success(f"🚨 Volume Spike Alert: {alert_msg}")
        
        # Price change alert
        if len(st.session_state.chart_data) > 1:
            previous_price = st.session_state.chart_data.iloc[-2]['close']
            price_change_pct = abs((current_price - previous_price) / previous_price)
            
            if price_change_pct > self.price_change_threshold:
                direction = "UP" if current_price > previous_price else "DOWN"
                alert_msg = f"Price movement {direction}: {price_change_pct*100:.1f}%"
                self.telegram.send_chart_alert("PRICE_CHANGE", security_id, current_price, current_volume, alert_msg)
                st.warning(f"📈 Price Change Alert: {alert_msg}")
    
    def create_candlestick_chart_with_footprint(self, data, show_volume_boxes=True):
        """Create candlestick chart with Pine Script volume footprint overlay"""
        if data is None or len(data) == 0:
            return None
            
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price Action with Volume Footprint [BigBeluga]', 'Volume'),
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
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['#26a69a' if close >= open else '#ef5350' 
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
        
        # Add Volume Footprint Boxes (Pine Script style)
        if show_volume_boxes:
            prices, volumes = self.volume_footprint_indicator.get_volume_profile_data()
            poc_data = self.volume_footprint_indicator.get_poc()
            
            if prices and volumes:
                max_volume = max(volumes)
                
                for i, (price, volume) in enumerate(zip(prices, volumes)):
                    # Calculate box width based on volume
                    width_ratio = volume / max_volume if max_volume > 0 else 0
                    
                    # Color based on POC
                    is_poc = poc_data and abs(price - poc_data['price']) < 0.01
                    color = self.volume_footprint_indicator.poc_color if is_poc else 'rgba(126, 132, 146, 0.5)'
                    
                    # Add volume footprint rectangles
                    step = (st.session_state.current_high - st.session_state.current_low) / self.volume_footprint_indicator.bins
                    
                    fig.add_shape(
                        type="rect",
                        x0=data['timestamp'].iloc[-1],
                        x1=data['timestamp'].iloc[-1] + pd.Timedelta(minutes=int(width_ratio * 60)),
                        y0=price,
                        y1=price + step,
                        fillcolor=color,
                        opacity=0.6,
                        line=dict(color=color, width=1),
                        row=1, col=1
                    )
        
        # Add POC line
        poc_data = self.volume_footprint_indicator.get_poc()
        if poc_data:
            fig.add_hline(
                y=poc_data['price'],
                line_dash="dash",
                line_color=self.volume_footprint_indicator.poc_color,
                line_width=2,
                annotation_text=f"POC: {poc_data['price']:.2f}",
                row=1, col=1
            )
        
        # Add session high/low lines (Pine Script style)
        if st.session_state.current_high and st.session_state.current_low:
            fig.add_hline(
                y=st.session_state.current_high,
                line_color="#26a69a",
                line_width=1,
                annotation_text=f"High: {st.session_state.current_high:.2f}",
                row=1, col=1
            )
            fig.add_hline(
                y=st.session_state.current_low,
                line_color="#ef5350", 
                line_width=1,
                annotation_text=f"Low: {st.session_state.current_low:.2f}",
                row=1, col=1
            )
        
        # Update layout with dark theme (TradingView style)
        fig.update_layout(
            title="DhanHQ Real-time Trading Dashboard with Volume Footprint [BigBeluga]",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template="plotly_dark",
            font=dict(color='white'),
            paper_bgcolor='#131722',
            plot_bgcolor='#131722'
        )
        
        return fig
    
    def create_volume_footprint_chart(self):
        """Create separate volume footprint visualization"""
        prices, volumes = self.volume_footprint_indicator.get_volume_profile_data()
        
        if not prices or not volumes:
            return None
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Get POC for highlighting
        poc_data = self.volume_footprint_indicator.get_poc()
        
        colors = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            is_poc = poc_data and abs(price - poc_data['price']) < 0.01
            colors.append(self.volume_footprint_indicator.poc_color if is_poc else 'rgba(126, 132, 146, 0.7)')
        
        fig.add_trace(go.Bar(
            x=volumes,
            y=prices,
            orientation='h',
            marker_color=colors,
            name='Volume Profile',
            text=[f'{v:.1f}' for v in volumes],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Volume Footprint Profile [BigBeluga]",
            xaxis_title="Volume",
            yaxis_title="Price Level",
            template="plotly_dark",
            height=400,
            font=dict(color='white'),
            paper_bgcolor='#131722',
            plot_bgcolor='#131722'
        )
        
        return fig
    
    def update_data(self, security_id, exchange_segment):
        """Update market data with rate limit handling and IST timezone"""
        last_api_call = st.session_state.get(f'last_api_call_{security_id}', datetime.min)
        time_since_last_call = (self.get_ist_time() - last_api_call).total_seconds()
        
        market_data = None
        
        # Only call API if enough time has passed (25+ seconds)
        if time_since_last_call >= 25:
            market_data = self.fetch_market_data(security_id, exchange_segment)
            st.session_state[f'last_api_call_{security_id}'] = self.get_ist_time()
            
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
                
                # Update volume footprint using Pine Script indicator
                self.update_volume_footprint_with_indicator(st.session_state.chart_data)
                
                # Check for alerts
                self.check_alerts(market_data, security_id)
                
                st.session_state.last_update = self.get_ist_time()
        
        # Return market data or latest historical data
        if market_data:
            return market_data
        elif len(st.session_state.chart_data) > 0:
            # Return the latest data point
            latest = st.session_state.chart_data.iloc[-1].to_dict()
            return latest
        
        return None

def main():
    st.title("📈 DhanHQ Trading Dashboard with Volume Footprint [BigBeluga] - IST")
    
    # Display current IST time
    current_ist = datetime.now(IST)
    st.sidebar.info(f"Current IST Time: {current_ist.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    st.sidebar.header("📊 Configuration")
    
    # Instrument selection
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
    
    # Trading session settings
    st.sidebar.header("⚡ Live Trading")
    update_interval = st.sidebar.slider("Update Interval (seconds)", 20, 60, 23)
    auto_refresh = st.sidebar.checkbox("Auto Refresh Live Data", value=True)
    
    # Data display options
    st.sidebar.header("📈 Data Display Options")
    data_mode = st.sidebar.radio(
        "Select Data Mode",
        ["Present Day Only", "Historical + Live", "Historical Only"],
        index=1
    )
    
    # Volume Footprint settings
    st.sidebar.header("📊 Volume Footprint [BigBeluga]")
    vf_bins = st.sidebar.slider("Volume Footprint Bins", 10, 40, 20)
    vf_timeframe = st.sidebar.selectbox(
        "Footprint Timeframe",
        ["1D", "2D", "3D", "4D", "5D", "1W", "2W", "3W", "1M", "2M", "3M"],
        index=0
    )
    show_dynamic_poc = st.sidebar.checkbox("Show Dynamic POC", value=False)
    show_volume_boxes = st.sidebar.checkbox("Show Volume Boxes", value=True)
    
    # Update volume footprint settings
    dashboard.volume_footprint_indicator.bins = vf_bins
    dashboard.volume_footprint_indicator.timeframe = vf_timeframe
    
    # Historical data section
    st.sidebar.header("📅 Historical Data")
    
    if data_mode == "Present Day Only":
        # Load only today's data
        historical_days = 1
        use_intraday = True
        st.sidebar.info("Loading present day data only")
    else:
        historical_days = st.sidebar.slider("Days of Historical Data", 1, 90, 7)
        use_intraday = st.sidebar.checkbox("Use Intraday Data (5min)", value=True)
    
    if st.sidebar.button("🔄 Reload Historical Data"):
        with st.spinner("Loading historical data..."):
            current_ist = dashboard.get_ist_time()
            
            if data_mode == "Present Day Only":
                # Today's data only
                from_date = current_ist.strftime("%Y-%m-%d 09:15:00")
                to_date = current_ist.strftime("%Y-%m-%d 15:30:00")
                interval = "5"
            else:
                # Multi-day data
                to_date = current_ist.strftime("%Y-%m-%d")
                from_date = (current_ist - timedelta(days=historical_days)).strftime("%Y-%m-%d")
                
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
                    dashboard.update_volume_footprint_with_indicator(historical_data)
                
                st.success(f"Loaded {len(historical_data)} data points for {selected_instrument_name}")
            else:
                st.error("Failed to load historical data")
    
    # Auto-load data on first run or instrument change
    if ('data_initialized' not in st.session_state or 
        st.session_state.get('last_instrument') != selected_instrument):
        
        st.session_state.data_initialized = True
        st.session_state.last_instrument = selected_instrument
        
        with st.spinner(f"Auto-loading data for {selected_instrument_name}..."):
            current_ist = dashboard.get_ist_time()
            
            if data_mode == "Present Day Only":
                from_date = current_ist.strftime("%Y-%m-%d 09:15:00")
                to_date = current_ist.strftime("%Y-%m-%d 15:30:00")
                interval = "5"
                days = 1
            else:
                days = 7  # Default 7 days
                to_date = current_ist.strftime("%Y-%m-%d 15:30:00")
                from_date = (current_ist - timedelta(days=days)).strftime("%Y-%m-%d 09:15:00")
                interval = "5"
            
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
                dashboard.update_volume_footprint_with_indicator(historical_data)
    
    # Market status
    current_ist_time = dashboard.get_ist_time()
    is_market_hours = (current_ist_time.weekday() < 5 and 
                      9 <= current_ist_time.hour < 16 and
                      not (current_ist_time.hour == 15 and current_ist_time.minute > 30))
    
    if is_market_hours:
        st.sidebar.success("🟢 Market is Open")
    else:
        st.sidebar.warning("🔴 Market is Closed")
    
    # Main update logic
    if st.sidebar.button("🔄 Manual Update") or (auto_refresh and data_mode != "Historical Only"):
        # Update with live data (if not historical only mode)
        if data_mode != "Historical Only":
            market_data = dashboard.update_data(security_id, exchange_segment)
        else:
            market_data = None
            if len(st.session_state.chart_data) > 0:
                market_data = st.session_state.chart_data.iloc[-1].to_dict()
        
        if market_data or len(st.session_state.chart_data) > 0:
            # Main chart layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if len(st.session_state.chart_data) > 0:
                    try:
                        # Filter data based on mode
                        display_data = st.session_state.chart_data.copy()
                        
                        if data_mode == "Present Day Only":
                            today = current_ist_time.date()
                            display_data = display_data[display_data['timestamp'].dt.date == today]
                        
                        chart = dashboard.create_candlestick_chart_with_footprint(
                            display_data, 
                            show_volume_boxes=show_volume_boxes
                        )
                        
                        if chart:
                            mode_text = f" ({data_mode})" if data_mode != "Historical + Live" else ""
                            chart.update_layout(title=f"{selected_instrument_name} - Volume Footprint [BigBeluga]{mode_text}")
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.error("Unable to create chart")
                    except Exception as e:
                        st.error(f"Chart error: {e}")
                else:
                    st.info("Loading chart data...")
            
            with col2:
                st.subheader("📊 Market Info")
                
                # Current/Latest price info
                if market_data and data_mode != "Historical Only":
                    st.metric("💰 Current Price", f"₹{market_data['close']:.2f}")
                    st.metric("📈 Volume", f"{market_data['volume']:,}")
                    st.metric("🟢 Buy Qty", f"{market_data.get('buy_quantity', 0):,}")
                    st.metric("🔴 Sell Qty", f"{market_data.get('sell_quantity', 0):,}")
                elif len(st.session_state.chart_data) > 0:
                    latest = st.session_state.chart_data.iloc[-1]
                    st.metric("💰 Last Price", f"₹{latest['close']:.2f}")
                    st.metric("📈 Volume", f"{latest['volume']:,}")
                    st.metric("🕐 Last Update", latest['timestamp'].strftime('%H:%M IST'))
                
                # Session High/Low
                if st.session_state.current_high and st.session_state.current_low:
                    st.metric("🔝 Session High", f"₹{st.session_state.current_high:.2f}")
                    st.metric("🔻 Session Low", f"₹{st.session_state.current_low:.2f}")
                
                # POC Info
                poc_data = dashboard.volume_footprint_indicator.get_poc()
                if poc_data:
                    st.metric("🎯 POC Price", f"₹{poc_data['price']:.2f}")
                    st.metric("📊 POC Volume", f"{poc_data['volume']:.1f}")
                
                # Volume Footprint Stats
                st.subheader("🔥 Volume Footprint")
                prices, volumes = dashboard.volume_footprint_indicator.get_volume_profile_data()
                if prices and volumes:
                    st.metric("📍 Price Levels", len(prices))
                    st.metric("📊 Total Volume", f"{sum(volumes):.1f}")
                    st.metric("⚙️ Bins", vf_bins)
                    st.metric("⏰ Timeframe", vf_timeframe)
            
            # Volume footprint chart
            st.subheader(f"🔥 Volume Footprint Profile [BigBeluga] - {selected_instrument_name}")
            footprint_chart = dashboard.create_volume_footprint_chart()
            if footprint_chart:
                st.plotly_chart(footprint_chart, use_container_width=True)
            else:
                st.info("Volume footprint will appear after accumulating more data.")
            
            # Data summary and recent data table
            col3, col4 = st.columns([1, 1])
            
            with col3:
                st.subheader(f"📈 Recent Data - {selected_instrument_name}")
                if len(st.session_state.chart_data) > 0:
                    display_data = st.session_state.chart_data.copy()
                    
                    if data_mode == "Present Day Only":
                        today = current_ist_time.date()
                        display_data = display_data[display_data['timestamp'].dt.date == today]
                    
                    recent_data = display_data.tail(10).copy()
                    recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%H:%M:%S IST')
                    recent_data = recent_data.round(2)
                    st.dataframe(recent_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']], 
                               use_container_width=True)
            
            with col4:
                st.subheader("📊 Data Statistics")
                if len(st.session_state.chart_data) > 0:
                    stats_data = st.session_state.chart_data.copy()
                    
                    if data_mode == "Present Day Only":
                        today = current_ist_time.date()
                        stats_data = stats_data[stats_data['timestamp'].dt.date == today]
                    
                    if len(stats_data) > 0:
                        st.metric("📊 Total Candles", len(stats_data))
                        st.metric("💹 Price Range", f"₹{stats_data['close'].max() - stats_data['close'].min():.2f}")
                        st.metric("🎯 Avg Volume", f"{stats_data['volume'].mean():,.0f}")
                        
                        data_start = stats_data['timestamp'].min()
                        data_end = stats_data['timestamp'].max()
                        st.metric("📅 Data Period", f"{data_start.strftime('%m/%d')} - {data_end.strftime('%m/%d')}")
            
            # Status information
            st.sidebar.success("✅ Connected to DhanHQ API")
            st.sidebar.info(f"🎯 Instrument: {selected_instrument_name}")
            st.sidebar.info(f"🆔 Security ID: {security_id}")
            st.sidebar.info(f"🏢 Exchange: {exchange_segment}")
            st.sidebar.info(f"📊 Data Mode: {data_mode}")
            
            if st.session_state.last_update:
                st.sidebar.info(f"🕐 Last Update: {st.session_state.last_update.strftime('%H:%M:%S IST')}")
            
            # Data status
            if len(st.session_state.chart_data) > 0:
                total_points = len(st.session_state.chart_data)
                
                if data_mode == "Present Day Only":
                    today = current_ist_time.date()
                    today_data = st.session_state.chart_data[st.session_state.chart_data['timestamp'].dt.date == today]
                    st.sidebar.metric("📈 Today's Data Points", len(today_data))
                else:
                    st.sidebar.metric("📈 Total Data Points", total_points)
                
                data_start = st.session_state.chart_data['timestamp'].min()
                data_end = st.session_state.chart_data['timestamp'].max()
                st.sidebar.info(f"📅 Data Range: {data_start.strftime('%m/%d %H:%M')} - {data_end.strftime('%m/%d %H:%M')} IST")
            
            # Volume Footprint status
            prices, volumes = dashboard.volume_footprint_indicator.get_volume_profile_data()
            if prices and volumes:
                st.sidebar.metric("🔥 Volume Levels", len(prices))
                st.sidebar.success("📊 Volume Footprint Active")
            
            # Auto-refresh functionality
            if auto_refresh and data_mode != "Historical Only":
                time.sleep(update_interval)
                st.rerun()
        else:
            st.error("Failed to fetch market data. Please check your API credentials.")
            st.info("Try loading historical data manually from the sidebar.")

if __name__ == "__main__":
    main()
