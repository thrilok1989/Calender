import streamlit as st
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
    
    def fetch_market_data(self, security_id="11536", exchange_segment="NSE_EQ"):
        """Fetch real-time market data using DhanHQ API"""
        try:
            quote_url = f"{self.base_url}/marketfeed/quote"
            quote_payload = {exchange_segment: [int(security_id)]}
            
            response = requests.post(quote_url, headers=self.headers, json=quote_payload)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and exchange_segment in data['data']:
                    market_data = data['data'][exchange_segment][security_id]
                    processed_data = self.process_market_data(market_data)
                    
                    if processed_data:
                        processed_data['security_id'] = security_id
                        processed_data['exchange_segment'] = exchange_segment
                        
                        # Store in database
                        self.db.insert_market_data(processed_data)
                    
                    return processed_data
            else:
                st.error(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Exception in API call: {e}")
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
                'buy_quantity': raw_data['buy_quantity'],
                'sell_quantity': raw_data['sell_quantity']
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
        """Update market data"""
        market_data = self.fetch_market_data(security_id, exchange_segment)
        
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
            
            return market_data
        
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
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    security_id = st.sidebar.text_input("Security ID", value="11536")
    exchange_segment = st.sidebar.selectbox(
        "Exchange Segment",
        ["NSE_EQ", "BSE_EQ", "NSE_FNO", "BSE_FNO", "MCX_FO"]
    )
    
    update_interval = st.sidebar.slider("Update Interval (seconds)", 20, 60, 23)
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    if st.sidebar.button("Manual Update") or auto_refresh:
        market_data = dashboard.update_data(security_id, exchange_segment)
        
        if market_data:
            # Main chart area
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if len(st.session_state.chart_data) > 0:
                    chart = dashboard.create_candlestick_chart(st.session_state.chart_data)
                    st.plotly_chart(chart, use_container_width=True)
            
            with col2:
                st.subheader("Market Info")
                st.metric("Current Price", f"₹{market_data['close']:.2f}")
                st.metric("Volume", f"{market_data['volume']:,}")
                st.metric("Buy Qty", f"{market_data['buy_quantity']:,}")
                st.metric("Sell Qty", f"{market_data['sell_quantity']:,}")
                
                if st.session_state.current_high and st.session_state.current_low:
                    st.metric("Session High", f"₹{st.session_state.current_high:.2f}")
                    st.metric("Session Low", f"₹{st.session_state.current_low:.2f}")
                
                poc_data = dashboard.calculate_poc()
                if poc_data:
                    st.metric("POC Price", f"₹{poc_data['price']:.2f}")
                    st.metric("POC Volume", f"{poc_data['volume']:.1f}")
            
            # Volume footprint chart
            st.subheader("Volume Footprint")
            footprint_chart = dashboard.create_volume_footprint_chart()
            if footprint_chart:
                st.plotly_chart(footprint_chart, use_container_width=True)
            
            # Recent data table
            st.subheader("Recent Market Data")
            if len(st.session_state.chart_data) > 0:
                recent_data = st.session_state.chart_data.tail(10).copy()
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%H:%M:%S')
                st.dataframe(recent_data)
            
            # Status info
            st.sidebar.success(f"✅ Connected to DhanHQ API")
            if st.session_state.last_update:
                st.sidebar.info(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
            
            # Auto-refresh functionality
            if auto_refresh:
                time.sleep(update_interval)
                st.rerun()

if __name__ == "__main__":
    main()
