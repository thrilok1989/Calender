import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import time
from datetime import datetime, timedelta
import json
from supabase import create_client, Client
import hashlib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Nifty Trading Chart",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for TradingView-like appearance
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stSelectbox > div > div > select {
        background-color: #1e1e1e;
        color: white;
    }
    .metric-container {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .price-up {
        color: #00ff88;
    }
    .price-down {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

class SupabaseDB:
    def __init__(self, url, key):
        self.client: Client = create_client(url, key)
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Test connection by trying to select from candle_data
            self.client.table('candle_data').select('id').limit(1).execute()
        except:
            st.info("Database tables may need to be created. Please run the SQL setup first.")
    
    def save_candle_data(self, symbol, exchange, timeframe, df):
        """Save candle data to Supabase with proper conflict handling"""
        if df.empty:
            return
        
        try:
            # Prepare data for insertion
            records = []
            for _, row in df.iterrows():
                record = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'timestamp': int(row['timestamp']),
                    'datetime': row['datetime'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                }
                records.append(record)
            
            # Insert data with conflict resolution on unique constraint
            response = self.client.table('candle_data').upsert(
                records,
                on_conflict='symbol,exchange,timeframe,timestamp'
            ).execute()
            
        except Exception as e:
            st.error(f"Error saving candle data: {str(e)}")
    
    def get_candle_data(self, symbol, exchange, timeframe, hours_back=24):
        """Retrieve candle data from Supabase"""
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - timedelta(hours=hours_back)
            
            result = self.client.table('candle_data')\
                .select('*')\
                .eq('symbol', symbol)\
                .eq('exchange', exchange)\
                .eq('timeframe', timeframe)\
                .gte('datetime', cutoff_time.isoformat())\
                .order('timestamp', desc=False)\
                .execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error retrieving candle data: {str(e)}")
            return pd.DataFrame()
    
    def save_user_preferences(self, user_id, timeframe, auto_refresh, days_back):
        """Save user preferences"""
        try:
            data = {
                'user_id': user_id,
                'timeframe': timeframe,
                'auto_refresh': auto_refresh,
                'days_back': days_back,
                'updated_at': datetime.now(pytz.UTC).isoformat()
            }
            
            self.client.table('user_preferences').upsert(data).execute()
            
        except Exception as e:
            st.error(f"Error saving preferences: {str(e)}")
    
    def get_user_preferences(self, user_id):
        """Get user preferences"""
        try:
            result = self.client.table('user_preferences')\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            
            if result.data:
                return result.data[0]
            else:
                return {
                    'timeframe': '5',
                    'auto_refresh': True,
                    'days_back': 1
                }
                
        except Exception as e:
            st.error(f"Error retrieving preferences: {str(e)}")
            return {'timeframe': '5', 'auto_refresh': True, 'days_back': 1}
    
    def save_market_analytics(self, symbol, analytics_data):
        """Save daily market analytics with proper conflict handling"""
        try:
            today = datetime.now(pytz.timezone('Asia/Kolkata')).date()
            
            data = {
                'symbol': symbol,
                'date': today.isoformat(),
                **analytics_data
            }
            
            # Insert with conflict resolution
            self.client.table('market_analytics').upsert(
                data,
                on_conflict='symbol,date'
            ).execute()
            
        except Exception as e:
            st.error(f"Error saving analytics: {str(e)}")
    
    def get_market_analytics(self, symbol, days_back=30):
        """Get historical market analytics"""
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days_back)
            
            result = self.client.table('market_analytics')\
                .select('*')\
                .eq('symbol', symbol)\
                .gte('date', cutoff_date.isoformat())\
                .order('date', desc=False)\
                .execute()
            
            if result.data:
                return pd.DataFrame(result.data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error retrieving analytics: {str(e)}")
            return pd.DataFrame()

class DhanAPI:
    def __init__(self, access_token, client_id):
        # Clean the tokens by stripping whitespace
        self.access_token = access_token.strip() if access_token else ""
        self.client_id = client_id.strip() if client_id else ""
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id
        }
        
    def get_intraday_data(self, security_id="13", exchange_segment="IDX_I", instrument="INDEX", interval="1", days_back=1):
        """Get intraday historical data"""
        url = f"{self.base_url}/charts/intraday"
        
        # Calculate date range
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": False,
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def get_ltp_data(self, security_id="13", exchange_segment="IDX_I"):
        """Get Last Traded Price with better error handling"""
        url = f"{self.base_url}/marketfeed/ltp"
        
        payload = {
            exchange_segment: [int(security_id)]
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"LTP API Error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.Timeout:
            st.error("LTP API request timed out")
            return None
        except Exception as e:
            st.error(f"Error fetching LTP: {str(e)}")
            return None

def process_candle_data(data, interval):
    """Process API response into DataFrame"""
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'timestamp': data['timestamp'],
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })
    
    # Convert timestamp to IST datetime
    ist = pytz.timezone('Asia/Kolkata')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
    
    return df

# === Pivot High/Low detection ===
def pivot_high(series, left, right):
    return series.shift(-right).rolling(left+right+1, center=True).max() == series.shift(-right)

def pivot_low(series, left, right):
    return series.shift(-right).rolling(left+right+1, center=True).min() == series.shift(-right)

# === Higher Timeframe Resampling ===
def resample_ohlc(df, tf):
    rule_map = {
        "240": "240min",
        "720": "720min",
        "D": "1D",
        "W": "1W"
    }
    rule = rule_map.get(tf, tf)
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

# === Pivot calculation ===
def get_pivots(df, tf="D", length=5):
    df_htf = resample_ohlc(df, tf)
    
    highs = df_htf['high']
    lows = df_htf['low']
    
    ph_mask = pivot_high(highs, length, length)
    pl_mask = pivot_low(lows, length, length)
    
    pivot_highs = highs[ph_mask].dropna()
    pivot_lows = lows[pl_mask].dropna()
    
    return pivot_highs, pivot_lows

def create_candlestick_chart(df, title, interval):
    """Create TradingView-style candlestick chart with pivot indicators"""
    if df.empty:
        return go.Figure()
    
    # Create subplots with correct parameters
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price', 'Volume'),
        row_width=[0.7, 0.3]  # 70% for price, 30% for volume
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Nifty 50',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        ),
        row=1, col=1
    )
    
    # Add pivot indicators if we have enough data
    if len(df) > 50:
        # Set datetime as index for pivot calculations
        df_indexed = df.set_index('datetime')
        
        # Calculate pivots for different timeframes
        configs = [
            ("240", 4, "green"),
            ("720", 5, "blue"),
            ("D",   5, "purple"),
        ]
        
        for tf, length, color in configs:
            try:
                ph, pl = get_pivots(df_indexed, tf, length)
                
                # Add pivot highs
                for t, v in ph.items():
                    fig.add_hline(
                        y=v, 
                        line_dash="dash", 
                        line_color=color, 
                        opacity=0.6,
                        row=1, col=1,
                        annotation_text=f"{tf} H {v:.1f}",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color=color
                    )
                
                # Add pivot lows
                for t, v in pl.items():
                    fig.add_hline(
                        y=v, 
                        line_dash="dash", 
                        line_color=color, 
                        opacity=0.6,
                        row=1, col=1,
                        annotation_text=f"{tf} L {v:.1f}",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color=color
                    )
            except Exception as e:
                # Skip if there's an error with this timeframe
                continue
    
    # Volume bars
    colors = ['#00ff88' if close >= open else '#ff4444' 
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
    
    # Update layout for TradingView-like appearance
    fig.update_layout(
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color='white'),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e'
    )
    
    # Update x-axis
    fig.update_xaxes(
        title_text="Time (IST)",
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        type='date'
    )
    
    # Update y-axis
    fig.update_yaxes(
        title_text="Price (₹)",
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        showgrid=True,
        gridwidth=1,
        gridcolor='#333333',
        row=2, col=1
    )
    
    return fig

def display_metrics(ltp_data, df, db, symbol="NIFTY50"):
    """Display price metrics and save analytics with fallbacks"""
    # Get current price with multiple fallbacks
    current_price = None
    
    # Try LTP API first
    if ltp_data and 'data' in ltp_data:
        for exchange, data in ltp_data['data'].items():
            for security_id, price_data in data.items():
                current_price = price_data.get('last_price')
                if current_price:
                    break
            if current_price:
                break
    
    # Fallback to latest close price from dataframe
    if current_price is None and not df.empty:
        current_price = df['close'].iloc[-1]
    
    # Final fallback
    if current_price is None:
        st.error("Could not retrieve current price")
        return
    
    # Calculate metrics
    if not df.empty and len(df) > 1:
        prev_close = df['close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        day_high = df['high'].max()
        day_low = df['low'].min()
        day_open = df['open'].iloc[0]
        volume = df['volume'].sum()
        avg_price = df['close'].mean()
        
        # Save analytics to database
        analytics_data = {
            'day_high': float(day_high),
            'day_low': float(day_low),
            'day_open': float(day_open),
            'day_close': float(current_price),
            'total_volume': int(volume),
            'avg_price': float(avg_price),
            'price_change': float(change),
            'price_change_pct': float(change_pct)
        }
        db.save_market_analytics(symbol, analytics_data)
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            color = "price-up" if change >= 0 else "price-down"
            st.markdown(f"""
            <div class="metric-container">
                <h4>Current Price</h4>
                <h2 class="{color}">₹{current_price:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "price-up" if change >= 0 else "price-down"
            sign = "+" if change >= 0 else ""
            st.markdown(f"""
            <div class="metric-container">
                <h4>Change</h4>
                <h3 class="{color}">{sign}{change:.2f} ({sign}{change_pct:.2f}%)</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h4>Day High</h4>
                <h3>₹{day_high:,.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h4>Day Low</h4>
                <h3>₹{day_low:,.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-container">
                <h4>Volume</h4>
                <h3>{volume:,}</h3>
            </div>
            """, unsafe_allow_html=True)

def validate_credentials(access_token, client_id):
    """Validate and clean API credentials"""
    issues = []
    
    # Clean tokens
    clean_token = access_token.strip() if access_token else ""
    clean_client_id = client_id.strip() if client_id else ""
    
    # Check for common issues
    if not clean_token:
        issues.append("Access token is empty")
    elif len(clean_token) < 50:
        issues.append("Access token seems too short")
    elif clean_token != access_token:
        issues.append("Access token has leading/trailing whitespace")
    
    if not clean_client_id:
        issues.append("Client ID is empty")
    elif len(clean_client_id) < 5:
        issues.append("Client ID seems too short")
    elif clean_client_id != client_id:
        issues.append("Client ID has leading/trailing whitespace")
    
    # Check for invalid characters
    if any(ord(c) < 32 or ord(c) > 126 for c in clean_token):
        issues.append("Access token contains invalid characters")
    
    if any(ord(c) < 32 or ord(c) > 126 for c in clean_client_id):
        issues.append("Client ID contains invalid characters")
    
    return clean_token, clean_client_id, issues

def get_user_id():
    """Generate a simple user ID based on session"""
    if 'user_id' not in st.session_state:
        # Create a simple user ID (in production, use proper authentication)
        st.session_state.user_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
    return st.session_state.user_id

def display_analytics_dashboard(db, symbol="NIFTY50"):
    """Display analytics dashboard"""
    st.subheader("Market Analytics Dashboard")
    
    analytics_df = db.get_market_analytics(symbol, days_back=30)
    
    if not analytics_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price trend chart
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=pd.to_datetime(analytics_df['date']),
                y=analytics_df['day_close'],
                mode='lines+markers',
                name='Close Price',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig_price.update_layout(
                title="30-Day Price Trend",
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            # Volume trend chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=pd.to_datetime(analytics_df['date']),
                y=analytics_df['total_volume'],
                name='Volume',
                marker_color='#4444ff'
            ))
            
            fig_volume.update_layout(
                title="30-Day Volume Trend",
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Summary statistics
        st.subheader("30-Day Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = analytics_df['day_close'].mean()
            st.metric("Average Price", f"₹{avg_price:,.2f}")
        
        with col2:
            volatility = analytics_df['price_change_pct'].std()
            st.metric("Volatility (σ)", f"{volatility:.2f}%")
        
        with col3:
            max_gain = analytics_df['price_change_pct'].max()
            st.metric("Max Daily Gain", f"{max_gain:.2f}%")
        
        with col4:
            max_loss = analytics_df['price_change_pct'].min()
            st.metric("Max Daily Loss", f"{max_loss:.2f}%")

def main():
    try:
        st.title("📈 Nifty 50 Trading Chart with Database")
        
        # Initialize Supabase
        try:
            supabase_url = st.secrets["supabase"]["url"]
            supabase_key = st.secrets["supabase"]["anon_key"]
            db = SupabaseDB(supabase_url, supabase_key)
            db.create_tables()
        except:
            st.error("Please configure your Supabase credentials in Streamlit secrets")
            st.info("""
            Add the following to your Streamlit secrets:
            ```
            [supabase]
            url = "your_supabase_url"
            anon_key = "your_supabase_anon_key"
            ```
            """)
            return
        
        # Initialize API credentials
        try:
            raw_access_token = st.secrets["dhan"]["access_token"]
            raw_client_id = st.secrets["dhan"]["client_id"]
            
            # Validate and clean credentials
            access_token, client_id, issues = validate_credentials(raw_access_token, raw_client_id)
            
            if issues:
                st.error("Issues found with API credentials:")
                for issue in issues:
                    st.error(f"• {issue}")
                
                # Show the raw vs cleaned values for debugging
                st.info("Raw values vs cleaned values:")
                st.code(f"Access token: '{raw_access_token}' -> '{access_token}'")
                st.code(f"Client ID: '{raw_client_id}' -> '{client_id}'")
                st.info("The app will try to use the cleaned values automatically.")
            
            st.sidebar.success("API credentials processed")
            
        except Exception as e:
            st.error("Please configure your Dhan API credentials in Streamlit secrets")
            st.error(f"Error: {str(e)}")
            return
        
        # Get user ID and preferences
        user_id = get_user_id()
        user_prefs = db.get_user_preferences(user_id)
        
        # Sidebar for configuration
        st.sidebar.header("Configuration")
        
        # Timeframe selection
        timeframes = {
            "1 min": "1",
            "3 min": "3", 
            "5 min": "5",
            "10 min": "10",
            "15 min": "15"
        }
        
        default_timeframe = next((k for k, v in timeframes.items() if v == user_prefs['timeframe']), "5 min")
        selected_timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            list(timeframes.keys()),
            index=list(timeframes.keys()).index(default_timeframe) if default_timeframe in timeframes else 2
        )
        
        interval = timeframes[selected_timeframe]
        
        # Auto-refresh settings
        auto_refresh = st.sidebar.checkbox("Auto Refresh (2 min)", value=user_prefs['auto_refresh'])
        
        # Days back for data
        days_back = st.sidebar.slider("Days of Historical Data", 1, 5, user_prefs['days_back'])
        
        # Data source preference
        use_cache = st.sidebar.checkbox("Use Cached Data", value=True, help="Use database cache for faster loading")
        
        # Pivot indicator settings
        st.sidebar.subheader("Pivot Indicator Settings")
        show_pivots = st.sidebar.checkbox("Show Pivot Levels", value=True)
        pivot_tf = st.sidebar.selectbox(
            "Pivot Timeframe",
            ["240min (4H)", "720min (12H)", "Daily", "All"],
            index=3
        )
        
        # Save preferences
        if st.sidebar.button("💾 Save Preferences"):
            db.save_user_preferences(user_id, interval, auto_refresh, days_back)
            st.sidebar.success("Preferences saved!")
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.experimental_rerun()
        
        # Show analytics dashboard
        show_analytics = st.sidebar.checkbox("Show Analytics Dashboard", value=False)
        
        # Initialize API
        api = DhanAPI(access_token, client_id)
        
        # Create tabs
        tab1, tab2 = st.tabs(["Live Chart", "Analytics"])
        
        with tab1:
            # Create placeholder for chart
            chart_container = st.container()
            metrics_container = st.container()
            
            # Data fetching strategy
            df = pd.DataFrame()
            
            if use_cache:
                # Try to get recent data from database first
                df = db.get_candle_data("NIFTY50", "IDX_I", interval, hours_back=days_back*24)
                
                # If no recent data or data is old, fetch from API
                if df.empty or (datetime.now(pytz.UTC) - df['datetime'].max().tz_convert(pytz.UTC)).total_seconds() > 300:
                    with st.spinner("Fetching latest data from API..."):
                        data = api.get_intraday_data(
                            security_id="13",
                            exchange_segment="IDX_I", 
                            instrument="INDEX",
                            interval=interval,
                            days_back=days_back
                        )
                        
                        if data:
                            df = process_candle_data(data, interval)
                            # Save to database
                            db.save_candle_data("NIFTY50", "IDX_I", interval, df)
            else:
                # Always fetch fresh data from API
                with st.spinner("Fetching fresh data from API..."):
                    data = api.get_intraday_data(
                        security_id="13",
                        exchange_segment="IDX_I", 
                        instrument="INDEX",
                        interval=interval,
                        days_back=days_back
                    )
                    
                    if data:
                        df = process_candle_data(data, interval)
                        # Save to database
                        db.save_candle_data("NIFTY50", "IDX_I", interval, df)
            
            # Get LTP data
            ltp_data = api.get_ltp_data("13", "IDX_I")
            
            # Display metrics
            with metrics_container:
                if not df.empty:
                    display_metrics(ltp_data, df, db)
            
            # Create and display chart
            with chart_container:
                if not df.empty:
                    fig = create_candlestick_chart(
                        df, 
                        f"Nifty 50 - {selected_timeframe} Chart", 
                        interval
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"📊 Data Points: {len(df)}")
                    with col2:
                        latest_time = df['datetime'].max().strftime("%Y-%m-%d %H:%M:%S IST")
                        st.info(f"🕐 Latest: {latest_time}")
                    with col3:
                        data_source = "Database Cache" if use_cache else "Live API"
                        st.info(f"📡 Source: {data_source}")
                else:
                    st.error("No data available. Please check your API credentials and try again.")
        
        with tab2:
            if show_analytics:
                display_analytics_dashboard(db)
            else:
                st.info("Enable 'Show Analytics Dashboard' in the sidebar to view historical analytics.")
        
        # Show current time
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
        st.sidebar.info(f"Last Updated: {current_time}")
        
        # Auto-refresh logic (simplified for stability)
        if auto_refresh:
            st.sidebar.info("Auto-refresh enabled - page will refresh every 2 minutes")
            time.sleep(120)  # 2 minutes
            st.experimental_rerun()
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your credentials and try again.")
        
        # Add a button to reset the app
        if st.button("Reset Application"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
