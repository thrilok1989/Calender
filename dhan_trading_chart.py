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
from supabase import create_client, Client

# Streamlit Configuration
st.set_page_config(
    page_title="DhanHQ Trading Dashboard with Volume Footprint",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh during market hours
def is_market_hours():
    """Check if current time is within market hours (Monday-Friday 8:30 AM - 4:00 PM IST)"""
    now = datetime.now(timezone('Asia/Kolkata'))
    
    # Check if it's a weekday (0=Monday, 6=Sunday)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Market hours: 8:30 AM to 4:00 PM
    market_start = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_start <= now <= market_end

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    try:
        supabase_url = st.secrets.get("SUPABASE_URL")
        supabase_key = st.secrets.get("SUPABASE_KEY")
        if supabase_url and supabase_key:
            return create_client(supabase_url, supabase_key)
        return None
    except:
        st.error("Supabase credentials not configured. Add SUPABASE_URL and SUPABASE_KEY to secrets.")
        return None

supabase = init_supabase()

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
if 'current_poc' not in st.session_state:
    st.session_state.current_poc = None
if 'spot_price' not in st.session_state:
    st.session_state.spot_price = None

def get_telegram_config():
    """Get Telegram configuration"""
    try:
        return {
            "bot_token": st.secrets.get("TELEGRAM_BOT_TOKEN"),
            "chat_id": st.secrets.get("TELEGRAM_CHAT_ID")
        }
    except:
        return {"bot_token": None, "chat_id": None}

def send_telegram_notification(message):
    """Send Telegram notification"""
    telegram_config = get_telegram_config()
    if not telegram_config["bot_token"] or not telegram_config["chat_id"]:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
        data = {
            "chat_id": telegram_config["chat_id"],
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=5)
        return response.status_code == 200
    except:
        return False

def store_poc_data(poc_data):
    """Store POC data in Supabase"""
    if not supabase:
        return False
    
    try:
        supabase.table("poc_history").insert(poc_data).execute()
        return True
    except Exception as e:
        st.error(f"Failed to store POC data: {e}")
        return False

def store_proximity_alert(alert_data):
    """Store proximity alert in Supabase"""
    if not supabase:
        return False
    
    try:
        supabase.table("proximity_alerts").insert(alert_data).execute()
        return True
    except Exception as e:
        st.error(f"Failed to store proximity alert: {e}")
        return False

def get_poc_history():
    """Retrieve POC history from Supabase"""
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table("poc_history").select("*").order("created_at", desc=True).limit(50).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Failed to retrieve POC history: {e}")
        return pd.DataFrame()

def get_proximity_alerts():
    """Retrieve proximity alerts from Supabase"""
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table("proximity_alerts").select("*").order("created_at", desc=True).limit(20).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Failed to retrieve proximity alerts: {e}")
        return pd.DataFrame()

def delete_all_history():
    """Delete all history from Supabase tables"""
    if not supabase:
        return False
    
    try:
        # Delete POC history
        supabase.table("poc_history").delete().neq("id", 0).execute()
        # Delete proximity alerts
        supabase.table("proximity_alerts").delete().neq("id", 0).execute()
        # Delete any other trading data tables if they exist
        supabase.table("trading_data").delete().neq("id", 0).execute()
        return True
    except Exception as e:
        st.error(f"Failed to delete history: {e}")
        return False

def calculate_improved_poc(price_data, volume_data):
    """Calculate POC using VWAP-based approach with strength analysis"""
    if len(price_data) == 0 or len(volume_data) == 0:
        return None
    
    try:
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'high': [d.get('high', 0) for d in price_data],
            'low': [d.get('low', 0) for d in price_data],
            'close': [d.get('close', 0) for d in price_data],
            'volume': [d.get('volume', 0) for d in price_data]
        })
        
        if df.empty or df['volume'].sum() == 0:
            return None
        
        # Calculate typical price (HLC/3) for each candle
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP as POC proxy
        df['volume_price'] = df['typical_price'] * df['volume']
        total_volume_price = df['volume_price'].sum()
        total_volume = df['volume'].sum()
        
        poc_price = total_volume_price / total_volume if total_volume > 0 else 0
        
        # Analyze POC strength and characteristics
        poc_analysis = analyze_poc_strength(df, poc_price)
        
        return {
            'poc_price': poc_price,
            'total_volume': total_volume,
            'strength': poc_analysis['strength'],
            'strength_score': poc_analysis['strength_score'],
            'role': poc_analysis['role'],  # Support or Resistance
            'confidence': poc_analysis['confidence']
        }
        
    except Exception as e:
        st.error(f"POC calculation error: {e}")
        return None

def analyze_poc_strength(df, poc_price):
    """Analyze whether POC is getting stronger/weaker and if it's support/resistance"""
    try:
        # Calculate distance of each candle from POC
        df['poc_distance'] = abs(df['typical_price'] - poc_price)
        df['poc_percentage'] = df['poc_distance'] / poc_price * 100
        
        # Volume concentration analysis
        close_to_poc = df[df['poc_percentage'] <= 0.5]  # Within 0.5% of POC
        volume_at_poc = close_to_poc['volume'].sum()
        total_volume = df['volume'].sum()
        
        volume_concentration = volume_at_poc / total_volume if total_volume > 0 else 0
        
        # Price action analysis around POC
        above_poc = df[df['typical_price'] > poc_price]
        below_poc = df[df['typical_price'] <= poc_price]
        
        volume_above = above_poc['volume'].sum()
        volume_below = below_poc['volume'].sum()
        
        # Determine if POC is acting as support or resistance
        recent_candles = df.tail(10)  # Last 10 candles
        recent_closes = recent_candles['close']
        
        touches_above = len(recent_candles[recent_candles['low'] <= poc_price * 1.002])  # Within 0.2%
        touches_below = len(recent_candles[recent_candles['high'] >= poc_price * 0.998])  # Within 0.2%
        
        # Determine role (Support/Resistance)
        if touches_above > touches_below and recent_closes.mean() > poc_price:
            role = "Support"
        elif touches_below > touches_above and recent_closes.mean() < poc_price:
            role = "Resistance"
        else:
            role = "Neutral"
        
        # Calculate strength score (0-100)
        strength_factors = {
            'volume_concentration': volume_concentration * 40,  # 40% weight
            'price_rejection': min(touches_above + touches_below, 5) * 8,  # 40% weight (max 5 touches)
            'volume_balance': (1 - abs(volume_above - volume_below) / total_volume) * 20  # 20% weight
        }
        
        strength_score = sum(strength_factors.values())
        
        # Classify strength
        if strength_score >= 70:
            strength = "Very Strong"
        elif strength_score >= 50:
            strength = "Strong"
        elif strength_score >= 30:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        # Confidence level
        confidence = min(100, volume_concentration * 100 + (touches_above + touches_below) * 10)
        
        return {
            'strength': strength,
            'strength_score': round(strength_score, 2),
            'role': role,
            'confidence': round(confidence, 2),
            'volume_concentration': round(volume_concentration * 100, 2),
            'touches': touches_above + touches_below
        }
        
    except Exception as e:
        st.error(f"POC strength analysis error: {e}")
        return {
            'strength': 'Unknown',
            'strength_score': 0,
            'role': 'Neutral',
            'confidence': 0,
            'volume_concentration': 0,
            'touches': 0
        }

def check_poc_change(new_poc_data, current_poc, threshold=0.1):
    """Check if POC has changed significantly"""
    if not new_poc_data or current_poc is None:
        return True
    
    new_price = new_poc_data['poc_price']
    price_change_pct = abs(new_price - current_poc) / current_poc * 100
    return price_change_pct > threshold

def check_spot_poc_proximity(spot_price, poc_price, proximity_threshold=0.2):
    """Check if spot price is near POC"""
    if not spot_price or not poc_price:
        return False
    
    distance_pct = abs(spot_price - poc_price) / poc_price * 100
    return distance_pct <= proximity_threshold

def update_poc_analysis(price_data, current_spot_price):
    """Update POC analysis with Supabase storage"""
    poc_data = calculate_improved_poc(price_data, price_data)  # Using same data for both params
    
    if not poc_data:
        return None
    
    current_time = datetime.now(timezone('Asia/Kolkata'))
    
    # Check for POC change
    if check_poc_change(poc_data, st.session_state.current_poc):
        old_poc = st.session_state.current_poc
        st.session_state.current_poc = poc_data['poc_price']
        
        # Prepare data for Supabase storage
        poc_entry = {
            'timestamp': current_time.isoformat(),
            'old_poc': old_poc,
            'new_poc': poc_data['poc_price'],
            'strength': poc_data['strength'],
            'strength_score': poc_data['strength_score'],
            'role': poc_data['role'],
            'confidence': poc_data['confidence'],
            'total_volume': poc_data['total_volume'],
            'change_pct': ((poc_data['poc_price'] - old_poc) / old_poc * 100) if old_poc else 0,
            'created_at': current_time.isoformat()
        }
        
        # Store in Supabase
        store_poc_data(poc_entry)
        
        # Send notification
        if old_poc:
            change_direction = "UP" if poc_data['poc_price'] > old_poc else "DOWN"
            change_pct = abs(poc_entry['change_pct'])
            
            message = f"""
🎯 <b>POC CHANGED</b> {change_direction}
Old POC: {old_poc:.2f}
New POC: {poc_data['poc_price']:.2f}
Change: {change_pct:.2f}%
Strength: {poc_data['strength']} ({poc_data['strength_score']:.1f}/100)
Role: {poc_data['role']}
Confidence: {poc_data['confidence']:.1f}%
Volume: {poc_data['total_volume']:.0f}
Time: {current_time.strftime('%H:%M:%S')}
            """
        else:
            message = f"""
🎯 <b>POC CALCULATED</b>
POC: {poc_data['poc_price']:.2f}
Strength: {poc_data['strength']} ({poc_data['strength_score']:.1f}/100)
Role: {poc_data['role']}
Confidence: {poc_data['confidence']:.1f}%
Volume: {poc_data['total_volume']:.0f}
Time: {current_time.strftime('%H:%M:%S')}
            """
        
        send_telegram_notification(message.strip())
    
    # Check spot price proximity to POC
    if current_spot_price and check_spot_poc_proximity(current_spot_price, poc_data['poc_price']):
        # Check cooldown to prevent spam
        last_alert = getattr(st.session_state, 'last_proximity_alert', None)
        if not last_alert or (current_time - last_alert).seconds > 60:
            
            proximity_entry = {
                'timestamp': current_time.isoformat(),
                'spot_price': current_spot_price,
                'poc_price': poc_data['poc_price'],
                'distance': abs(current_spot_price - poc_data['poc_price']),
                'distance_pct': abs(current_spot_price - poc_data['poc_price']) / poc_data['poc_price'] * 100,
                'poc_strength': poc_data['strength'],
                'poc_role': poc_data['role'],
                'created_at': current_time.isoformat()
            }
            
            # Store in Supabase
            store_proximity_alert(proximity_entry)
            
            message = f"""
🔔 <b>SPOT NEAR POC</b> ({poc_data['role']})
Spot: {current_spot_price:.2f}
POC: {poc_data['poc_price']:.2f} ({poc_data['strength']})
Distance: {proximity_entry['distance']:.2f} ({proximity_entry['distance_pct']:.2f}%)
Strength: {poc_data['strength_score']:.1f}/100
Time: {current_time.strftime('%H:%M:%S')}
            """
            
            send_telegram_notification(message.strip())
            st.session_state.last_proximity_alert = current_time
    
    return poc_data

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
        
        # Market hours status
        market_open = is_market_hours()
        status_color = "🟢" if market_open else "🔴"
        st.sidebar.markdown(f"{status_color} **Market Status:** {'Open' if market_open else 'Closed'}")
        
        # Instrument selection
        selected_instrument = st.sidebar.selectbox(
            "Select Instrument",
            list(self.instruments.keys()),
            index=2  # Default to NIFTY INDEX
        )
        
        # Data mode - Live Only as default
        data_mode = st.sidebar.radio(
            "Select Data Mode",
            ["Live Only", "Historical Only", "Historical + Live"],
            index=0  # Default to Live Only
        )
        
        # Update interval - Auto-update during market hours
        if market_open:
            st.sidebar.info("Auto-updating every 25 seconds during market hours")
            update_interval = 25
        else:
            update_interval = st.sidebar.slider(
                "Manual Update Interval (seconds)",
                min_value=10,
                max_value=120,
                value=30
            )
        
        # Historical data settings (only show if needed)
        if data_mode in ["Historical Only", "Historical + Live"]:
            days_back = st.sidebar.slider(
                "Days of Historical Data",
                min_value=1,
                max_value=7,
                value=1
            )
            
            timeframe = st.sidebar.selectbox(
                "Timeframe",
                ["1", "5", "15", "25", "60"],
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
        
        # POC Settings
        st.sidebar.markdown("### POC Settings")
        poc_change_threshold = st.sidebar.slider(
            "POC Change Threshold (%)",
            min_value=0.05,
            max_value=1.0,
            value=0.1,
            step=0.05
        )
        
        poc_proximity_threshold = st.sidebar.slider(
            "POC Proximity Alert (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.05
        )
        
        return {
            "instrument": selected_instrument,
            "update_interval": update_interval,
            "data_mode": data_mode,
            "days_back": days_back,
            "timeframe": timeframe,
            "footprint_bins": footprint_bins,
            "market_open": market_open,
            "poc_change_threshold": poc_change_threshold,
            "poc_proximity_threshold": poc_proximity_threshold
        }
    
    def display_status(self):
        """Display API status and POC metrics"""
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
            if st.session_state.current_poc:
                st.metric("Current POC", f"{st.session_state.current_poc:.2f}")
            else:
                st.metric("Current POC", "Calculating...")
        
        with col4:
            if st.session_state.spot_price:
                st.metric("Spot Price", f"{st.session_state.spot_price:.2f}")
            else:
                st.metric("Spot Price", "Fetching...")
    
    def display_poc_analysis(self):
        """Display POC analysis and history"""
        st.markdown("## Point of Control (POC) Analysis")
        
        # Current POC Info
        if st.session_state.current_poc:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Current POC:** {st.session_state.current_poc:.2f}")
            
            with col2:
                if st.session_state.spot_price:
                    distance = abs(st.session_state.spot_price - st.session_state.current_poc)
                    st.info(f"**Distance from POC:** {distance:.2f}")
            
            with col3:
                if st.session_state.last_poc_notification:
                    last_change = st.session_state.last_poc_notification
                    time_since = datetime.now(timezone('Asia/Kolkata')) - last_change
                    st.info(f"**Last POC Change:** {time_since.seconds//60}m ago")
        
        # POC History
        if st.session_state.poc_history:
            st.markdown("### POC Change History")
            
            history_df = pd.DataFrame(st.session_state.poc_history)
            
            # Display recent changes
            if len(history_df) > 0:
                recent_changes = history_df.tail(10)[['timestamp', 'old_poc', 'new_poc', 'change_pct', 'poc_volume']]
                recent_changes.columns = ['Time', 'Old POC', 'New POC', 'Change %', 'Volume']
                st.dataframe(recent_changes, use_container_width=True)
        
        # Proximity Alerts
        if st.session_state.poc_proximity_alerts:
            st.markdown("### POC Proximity Alerts")
            
            alerts_df = pd.DataFrame(st.session_state.poc_proximity_alerts)
            
            if len(alerts_df) > 0:
                recent_alerts = alerts_df.tail(5)[['timestamp', 'spot_price', 'poc_price', 'distance']]
                recent_alerts.columns = ['Time', 'Spot Price', 'POC', 'Distance']
                st.dataframe(recent_alerts, use_container_width=True)
    
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
    """Main application function with enhanced POC analysis and Supabase integration"""
    st.title("DhanHQ Trading Dashboard with Enhanced POC Analysis")
    
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
        
        2. Add Supabase credentials for data storage:
        ```
        SUPABASE_URL = "your_supabase_url"
        SUPABASE_KEY = "your_supabase_anon_key"
        ```
        
        3. Optional - Add Telegram notifications:
        ```
        TELEGRAM_BOT_TOKEN = "your_bot_token"
        TELEGRAM_CHAT_ID = "your_chat_id"
        ```
        """)
        return
    
    if not supabase:
        st.error("Supabase connection failed. Check your SUPABASE_URL and SUPABASE_KEY.")
        return
    
    current_time = datetime.now(timezone('Asia/Kolkata'))
    st.markdown(f"**Current IST Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    market_open = is_market_hours()
    
    dashboard = TradingDashboard()
    
    config = dashboard.display_configuration()
    
    dashboard.display_status()
    
    should_fetch_data = False
    
    if market_open and config["data_mode"] in ["Live Only", "Historical + Live"]:
        if (st.session_state.last_update is None or 
            (current_time - st.session_state.last_update).seconds >= 25):
            should_fetch_data = True
        
        if st.session_state.last_update:
            seconds_since_update = (current_time - st.session_state.last_update).seconds
            seconds_until_next = 25 - (seconds_since_update % 25)
            st.info(f"Next auto-update in: {seconds_until_next} seconds")
    
    manual_fetch = st.sidebar.button("Fetch Data Now")
    
    if should_fetch_data or manual_fetch:
        with st.spinner("Fetching live data and updating enhanced POC analysis..."):
            dashboard.fetch_and_update_data(config)
            
            analysis_data = st.session_state.live_data if not st.session_state.live_data.empty else st.session_state.historical_data
            
            if not analysis_data.empty:
                latest_price = analysis_data['close'].iloc[-1]
                st.session_state.spot_price = latest_price
                
                price_data_list = analysis_data.to_dict('records')
                poc_result = update_poc_analysis(price_data_list, st.session_state.spot_price)
                
                if poc_result:
                    strength_indicator = ""
                    if poc_result['strength'] == "Very Strong":
                        strength_indicator = "🟢"
                    elif poc_result['strength'] == "Strong":
                        strength_indicator = "🟡"
                    elif poc_result['strength'] == "Moderate":
                        strength_indicator = "🟠"
                    else:
                        strength_indicator = "🔴"
                    
                    role_indicator = "🛡️" if poc_result['role'] == "Support" else "🚧" if poc_result['role'] == "Resistance" else "⚖️"
                    
                    st.success(f"{strength_indicator} POC: {poc_result['poc_price']:.2f} | Strength: {poc_result['strength']} ({poc_result['strength_score']:.1f}/100) | {role_indicator} Role: {poc_result['role']}")
        
        if market_open and not manual_fetch:
            time.sleep(1)
            st.rerun()
    
    dashboard.display_poc_analysis()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Export")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Export JSON", help="Export all data including POC analysis"):
            poc_history_df = get_poc_history()
            proximity_alerts_df = get_proximity_alerts()
            
            export_data = {
                'export_timestamp': current_time.isoformat(),
                'current_poc': st.session_state.current_poc,
                'current_spot': st.session_state.spot_price,
                'historical_data': st.session_state.historical_data.to_dict('records') if not st.session_state.historical_data.empty else [],
                'live_data': st.session_state.live_data.to_dict('records') if not st.session_state.live_data.empty else [],
                'poc_history': poc_history_df.to_dict('records') if not poc_history_df.empty else [],
                'proximity_alerts': proximity_alerts_df.to_dict('records') if not proximity_alerts_df.empty else []
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            st.sidebar.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"enhanced_trading_data_{current_time.strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export CSV", help="Export price data as CSV"):
            if not st.session_state.live_data.empty:
                csv_data = st.session_state.live_data.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"live_data_{current_time.strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.sidebar.warning("No live data to export")

# Auto-refresh during market hours
if is_market_hours():
    time.sleep(0.5)
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    st.session_state.refresh_counter += 1
    
    if st.session_state.refresh_counter % 50 == 0:
        st.rerun()

if __name__ == "__main__":
    main().session_state.refresh_counter = 0
    
    st.session_state.refresh_counter += 1
    
    if st.session_state.refresh_counter % 50 == 0:
        st.rerun()

if __name__ == "__main__":
    main()DhanHQ credentials not configured!")
        st.markdown("""
        ### Setup Instructions:
        1. Add your DhanHQ credentials to Streamlit secrets:
        ```
        DHAN_CLIENT_ID = "your_client_id"
        DHAN_ACCESS_TOKEN = "your_access_token"
        ```
        
        2. Add Supabase credentials for data storage:
        ```
        SUPABASE_URL = "your_supabase_url"
        SUPABASE_KEY = "your_supabase_anon_key"
        ```
        
        3. Optional - Add Telegram notifications:
        ```
        TELEGRAM_BOT_TOKEN = "your_bot_token"
        TELEGRAM_CHAT_ID = "your_chat_id"
        ```
        
        ### Required Supabase Tables:
        Create these tables in your Supabase database:
        
        **poc_history table:**
        ```sql
        CREATE TABLE poc_history (
            id SERIAL PRIMARY KEY,
            timestamp TEXT,
            old_poc REAL,
            new_poc REAL,
            strength TEXT,
            strength_score REAL,
            role TEXT,
            confidence REAL,
            total_volume REAL,
            change_pct REAL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        ```
        
        **proximity_alerts table:**
        ```sql
        CREATE TABLE proximity_alerts (
            id SERIAL PRIMARY KEY,
            timestamp TEXT,
            spot_price REAL,
            poc_price REAL,
            distance REAL,
            distance_pct REAL,
            poc_strength TEXT,
            poc_role TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        ```
        
        **trading_data table (optional):**
        ```sql
        CREATE TABLE trading_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ,
            instrument TEXT,
            price_data JSONB,
            volume_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        ```
        """)
        return
    
    # Check Supabase connection
    if not supabase:
        st.error("Supabase connection failed. Check your SUPABASE_URL and SUPABASE_KEY.")
        return
    
    # Current time display
    current_time = datetime.now(timezone('Asia/Kolkata'))
    st.markdown(f"**Current IST Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Market hours check
    market_open = is_market_hours()
    
    # Initialize dashboard
    dashboard = TradingDashboard()
    
    # Display configuration
    config = dashboard.display_configuration()
    
    # Display status
    dashboard.display_status()
    
    # Auto-refresh logic for market hours
    should_fetch_data = False
    
    if market_open and config["data_mode"] in ["Live Only", "Historical + Live"]:
        # Auto-fetch during market hours every 25 seconds
        if (st.session_state.last_update is None or 
            (current_time - st.session_state.last_update).seconds >= 25):
            should_fetch_data = True
        
        # Show countdown timer
        if st.session_state.last_update:
            seconds_since_update = (current_time - st.session_state.last_update).seconds
            seconds_until_next = 25 - (seconds_since_update % 25)
            st.info(f"Next auto-update in: {seconds_until_next} seconds")
    
    # Manual fetch button
    manual_fetch = st.sidebar.button("Fetch Data Now")
    
    if should_fetch_data or manual_fetch:
        with st.spinner("Fetching live data and updating enhanced POC analysis..."):
            # Fetch and update data
            dashboard.fetch_and_update_data(config)
            
            # Update volume footprint analysis
            analysis_data = st.session_state.live_data if not st.session_state.live_data.empty else st.session_state.historical_data
            
            if not analysis_data.empty:
                dashboard.update_volume_footprint_with_indicator(analysis_data)
                
                # Update spot price
                latest_price = analysis_data['close'].iloc[-1]
                st.session_state.spot_price = latest_price
                
                # Enhanced POC Analysis with strength detection
                price_data_list = analysis_data.to_dict('records')
                poc_result = update_poc_analysis(price_data_list, st.session_state.spot_price)
                
                if poc_result:
                    strength_indicator = ""
                    if poc_result['strength'] == "Very Strong":
                        strength_indicator = "🟢"
                    elif poc_result['strength'] == "Strong":
                        strength_indicator = "🟡"
                    elif poc_result['strength'] == "Moderate":
                        strength_indicator = "🟠"
                    else:
                        strength_indicator = "🔴"
                    
                    role_indicator = "🛡️" if poc_result['role'] == "Support" else "🚧" if poc_result['role'] == "Resistance" else "⚖️"
                    
                    st.success(f"{strength_indicator} POC: {poc_result['poc_price']:.2f} | Strength: {poc_result['strength']} ({poc_result['strength_score']:.1f}/100) | {role_indicator} Role: {poc_result['role']}")
        
        # Auto-refresh page during market hours
        if market_open and not manual_fetch:
            time.sleep(1)
            st.rerun()
    
    # Display Enhanced POC Analysis
    dashboard.display_poc_analysis()
    
    # Display charts with POC overlay
    poc_history_df = get_poc_history()
    current_poc_data = poc_history_df.iloc[0].to_dict() if not poc_history_df.empty else None
    dashboard.display_enhanced_charts(current_poc_data)
    
    # Export functionality with enhanced data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Export")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Export JSON", help="Export all data including POC analysis"):
            poc_history_df = get_poc_history()
            proximity_alerts_df = get_proximity_alerts()
            
            export_data = {
                'export_timestamp': current_time.isoformat(),
                'current_poc': st.session_state.current_poc,
                'current_spot': st.session_state.spot_price,
                'historical_data': st.session_state.historical_data.to_dict('records') if not st.session_state.historical_data.empty else [],
                'live_data': st.session_state.live_data.to_dict('records') if not st.session_state.live_data.empty else [],
                'poc_history': poc_history_df.to_dict('records') if not poc_history_df.empty else [],
                'proximity_alerts': proximity_alerts_df.to_dict('records') if not proximity_alerts_df.empty else []
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            st.sidebar.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"enhanced_trading_data_{current_time.strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export CSV", help="Export price data as CSV"):
            if not st.session_state.historical_data.empty:
                csv_data = st.session_state.historical_data.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"price_data_{current_time.strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.sidebar.warning("No price data to export")

# Auto-refresh during market hours
if is_market_hours():
    # This creates a 25-second auto-refresh during market hours
    time.sleep(0.5)
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    st.session_state.refresh_counter += 1
    
    # Trigger refresh approximately every 25 seconds
    if st.session_state.refresh_counter % 50 == 0:  # Adjust this number based on actual timing
        st.rerun()

if __name__ == "__main__":
    main()