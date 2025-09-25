import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import numpy as np
import math
from scipy.stats import norm
from datetime import datetime, timedelta
import json
import pyotp

# Page config
st.set_page_config(page_title="Nifty Analyzer", page_icon="📈", layout="wide")

# Function to check if it's market hours
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if it's a weekday (Monday to Friday)
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Check if current time is between 9:00 AM and 3:45 PM IST
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    
    return market_start <= now <= market_end

# Only run autorefresh during market hours
if is_market_hours():
    st_autorefresh(interval=35000, key="refresh")
else:
    st.info("Market is closed. Auto-refresh disabled.")

# Credentials
ANGEL_CLIENT_CODE = st.secrets.get("ANGEL_CLIENT_CODE", "")
ANGEL_PIN = st.secrets.get("ANGEL_PIN", "")
ANGEL_TOTP = st.secrets.get("ANGEL_TOTP", "")
ANGEL_API_KEY = st.secrets.get("ANGEL_API_KEY", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))

# Angel One Constants
NIFTY_TOKEN = "99926000"  # Nifty token from master data
NIFTY_EXCHANGE = "NSE"

class AngelOneAPI:
    def __init__(self):
        self.base_url = "https://apiconnect.angelone.in"
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '127.0.0.1',
            'X-ClientPublicIP': '127.0.0.1',
            'X-MACAddress': '00:00:00:00:00:00',
            'X-PrivateKey': ANGEL_API_KEY
        }
        self.authenticated = False
    
    def generate_totp(self):
        """Generate TOTP code from secret"""
        if not ANGEL_TOTP:
            return ""
        try:
            totp = pyotp.TOTP(ANGEL_TOTP)
            return totp.now()
        except:
            return ""
        
    def login(self):
        """Authenticate with Angel One API"""
        if not ANGEL_CLIENT_CODE or not ANGEL_PIN or not ANGEL_API_KEY:
            st.error("Missing Angel One credentials in secrets")
            return False
            
        url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
        
        payload = {
            "clientcode": ANGEL_CLIENT_CODE,
            "password": ANGEL_PIN,
            "totp": self.generate_totp(),
            "state": "live"
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            data = response.json()
            
            if data.get('status') and data.get('data'):
                self.jwt_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = data['data']['feedToken']
                self.authenticated = True
                return True
            else:
                st.error(f"Login failed: {data.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            st.error(f"Login error: {str(e)}")
            return False
    
    def get_auth_headers(self):
        """Get headers with authorization"""
        headers = self.headers.copy()
        if self.jwt_token:
            headers['Authorization'] = f'Bearer {self.jwt_token}'
        return headers
    
    def get_historical_data(self, interval="FIVE_MINUTE", days_back=1):
        """Get historical candlestick data"""
        if not self.authenticated and not self.login():
            return None
            
        url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
        
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "exchange": NIFTY_EXCHANGE,
            "symboltoken": NIFTY_TOKEN,
            "interval": interval,
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        }
        
        try:
            response = requests.post(url, headers=self.get_auth_headers(), json=payload)
            data = response.json()
            
            if data.get('status') and data.get('data'):
                return data['data']
            else:
                st.error(f"Historical data error: {data.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            st.error(f"Historical data error: {str(e)}")
            return None
    
    def get_ltp_data(self):
        """Get Last Traded Price"""
        if not self.authenticated and not self.login():
            return None
            
        url = f"{self.base_url}/rest/secure/angelbroking/market/v1/quote/"
        
        payload = {
            "mode": "LTP",
            "exchangeTokens": {
                NIFTY_EXCHANGE: [NIFTY_TOKEN]
            }
        }
        
        try:
            response = requests.post(url, headers=self.get_auth_headers(), json=payload)
            data = response.json()
            
            if data.get('status') and data.get('data', {}).get('fetched'):
                return data['data']['fetched'][0]
            else:
                return None
        except Exception as e:
            st.error(f"LTP error: {str(e)}")
            return None

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def process_candle_data(data):
    """Process Angel One historical data format: [timestamp, open, high, low, close, volume]"""
    if not data:
        return pd.DataFrame()
    
    df_data = []
    for candle in data:
        if len(candle) >= 6:
            df_data.append({
                'timestamp': pd.to_datetime(candle[0]),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': int(candle[5]) if candle[5] else 0
            })
    
    if not df_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(df_data)
    ist = pytz.timezone('Asia/Kolkata')
    df['datetime'] = pd.to_datetime(df['timestamp']).dt.tz_convert(ist)
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['close'].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_pivots(df, timeframe="FIVE_MINUTE", length=4):
    """Get pivot points from price data"""
    if df.empty:
        return []
    
    # Map timeframe to pandas resample rule
    rule_map = {
        "THREE_MINUTE": "3min", 
        "FIVE_MINUTE": "5min", 
        "TEN_MINUTE": "10min", 
        "FIFTEEN_MINUTE": "15min"
    }
    rule = rule_map.get(timeframe, "5min")
    
    df_temp = df.set_index('datetime')
    try:
        resampled = df_temp.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(resampled) < length * 2 + 1:
            return []
        
        max_vals = resampled['high'].rolling(window=length*2+1, center=True).max()
        min_vals = resampled['low'].rolling(window=length*2+1, center=True).min()
        
        pivots = []
        for timestamp, value in resampled['high'][resampled['high'] == max_vals].items():
            pivots.append({'type': 'high', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        
        for timestamp, value in resampled['low'][resampled['low'] == min_vals].items():
            pivots.append({'type': 'low', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        
        return pivots
    except:
        return []

def create_chart(df, title):
    """Create candlestick chart with RSI and volume"""
    if df.empty:
        return go.Figure()
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df)
    
    # Create subplots with 3 rows (price, volume, RSI)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price', 'Volume', 'RSI')
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Volume chart
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)
    
    # RSI chart
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['rsi'], name='RSI',
        line=dict(color='#ff9900', width=2)
    ), row=3, col=1)
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # Add pivot levels
    if len(df) > 50:
        timeframes = ["FIVE_MINUTE", "TEN_MINUTE", "FIFTEEN_MINUTE"]
        colors = ["#ff9900", "#ff44ff", '#4444ff']
        
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            x_start, x_end = df['datetime'].min(), df['datetime'].max()
            
            for pivot in pivots[-5:]:
                fig.add_shape(type="line", x0=x_start, x1=x_end,
                            y0=pivot['value'], y1=pivot['value'],
                            line=dict(color=color, width=1, dash="dash"), row=1, col=1)
    
    fig.update_layout(title=title, template='plotly_dark', height=800,
                     xaxis_rangeslider_visible=False, showlegend=False)
    
    # Update y-axis for RSI
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    return fig

def create_mock_options_data(current_price):
    """Create mock options data for demonstration (replace with actual Angel One options API when available)"""
    if not current_price:
        return None
    
    # Round to nearest 50 for ATM strike
    atm_strike = round(current_price / 50) * 50
    
    strikes = [atm_strike - 100, atm_strike - 50, atm_strike, atm_strike + 50, atm_strike + 100]
    
    options_data = []
    for strike in strikes:
        # Mock data - replace with actual options chain API
        ce_oi = np.random.randint(10000, 100000)
        pe_oi = np.random.randint(10000, 100000)
        ce_chg_oi = np.random.randint(-5000, 5000)
        pe_chg_oi = np.random.randint(-5000, 5000)
        
        zone = 'ATM' if strike == atm_strike else 'ITM' if strike < current_price else 'OTM'
        level = "Support" if pe_oi > 1.12 * ce_oi else "Resistance" if ce_oi > 1.12 * pe_oi else "Neutral"
        
        chg_oi_bias = "Bullish" if ce_chg_oi < pe_chg_oi else "Bearish"
        volume_bias = "Bullish" if np.random.random() > 0.5 else "Bearish"
        ask_bias = "Bullish" if np.random.random() > 0.5 else "Bearish"
        bid_bias = "Bullish" if np.random.random() > 0.5 else "Bearish"
        
        options_data.append({
            "Strike": strike,
            "Zone": zone,
            "Level": level,
            "ChgOI_Bias": chg_oi_bias,
            "Volume_Bias": volume_bias,
            "Ask_Bias": ask_bias,
            "Bid_Bias": bid_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else 0, 2),
            "changeinOpenInterest_CE": ce_chg_oi,
            "changeinOpenInterest_PE": pe_chg_oi
        })
    
    return pd.DataFrame(options_data)

def check_signals(df, option_data, current_price, proximity=5):
    """Check for trading signals"""
    if df.empty or option_data is None or not current_price:
        return
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df)
    current_rsi = df['rsi'].iloc[-1] if not df.empty else None
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return
    
    row = atm_data.iloc[0]
    
    ce_chg_oi = abs(row.get('changeinOpenInterest_CE', 0))
    pe_chg_oi = abs(row.get('changeinOpenInterest_PE', 0))
    
    bias_aligned_bullish = (
        row['ChgOI_Bias'] == 'Bullish' and 
        row['Volume_Bias'] == 'Bullish' and
        row['Ask_Bias'] == 'Bullish' and
        row['Bid_Bias'] == 'Bullish'
    )
    
    bias_aligned_bearish = (
        row['ChgOI_Bias'] == 'Bearish' and 
        row['Volume_Bias'] == 'Bearish' and
        row['Ask_Bias'] == 'Bearish' and
        row['Bid_Bias'] == 'Bearish'
    )
    
    # PRIMARY SIGNAL
    pivots = get_pivots(df, "FIVE_MINUTE") + get_pivots(df, "TEN_MINUTE") + get_pivots(df, "FIFTEEN_MINUTE")
    near_pivot = False
    pivot_level = None
    
    for pivot in pivots:
        if abs(current_price - pivot['value']) <= proximity:
            near_pivot = True
            pivot_level = pivot
            break
    
    if near_pivot:
        primary_bullish_signal = (row['Level'] == 'Support' and bias_aligned_bullish)
        primary_bearish_signal = (row['Level'] == 'Resistance' and bias_aligned_bearish)
        
        if primary_bullish_signal or primary_bearish_signal:
            signal_type = "CALL" if primary_bullish_signal else "PUT"
            price_diff = current_price - pivot_level['value']
            
            message = f"""
🚨 PRIMARY NIFTY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']} at ₹{pivot_level['value']:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.2f}

Conditions: {row['Level']}, All Bias Aligned
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success(f"🔔 PRIMARY {signal_type} signal sent!")

def main():
    st.title("📈 Nifty Trading Analyzer (Angel One)")
    
    # Show market status
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"⚠️ Market is closed. Current time: {current_time.strftime('%H:%M:%S IST')}")
        st.info("Market hours: Monday-Friday, 9:00 AM to 3:45 PM IST")
    
    st.sidebar.header("Settings")
    interval_map = {"1min": "ONE_MINUTE", "3min": "THREE_MINUTE", "5min": "FIVE_MINUTE", 
                   "10min": "TEN_MINUTE", "15min": "FIFTEEN_MINUTE"}
    interval_display = st.sidebar.selectbox("Timeframe", list(interval_map.keys()), index=2)
    interval = interval_map[interval_display]
    
    proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
    enable_signals = st.sidebar.checkbox("Enable Signals", value=True)
    
    # Initialize API
    api = AngelOneAPI()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chart")
        
        # Get historical data
        data = api.get_historical_data(interval)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        # Get current price
        ltp_data = api.get_ltp_data()
        current_price = ltp_data.get('ltp', 0) if ltp_data else None
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        if not df.empty and len(df) > 1:
            prev_close = df['close'].iloc[-2]
            change = current_price - prev_close if current_price else 0
            change_pct = (change / prev_close) * 100 if prev_close else 0
            
            # Calculate RSI
            current_rsi = None
            if not df.empty:
                df['rsi'] = calculate_rsi(df)
                current_rsi = df['rsi'].iloc[-1]
            
            col1_m, col2_m, col3_m, col4_m = st.columns(4)
            with col1_m:
                st.metric("Price", f"₹{current_price:,.2f}" if current_price else "N/A", 
                         f"{change:+.2f} ({change_pct:+.2f}%)" if current_price else None)
            with col2_m:
                st.metric("High", f"₹{df['high'].max():,.2f}")
            with col3_m:
                st.metric("Low", f"₹{df['low'].min():,.2f}")
            with col4_m:
                st.metric("RSI", f"{current_rsi:.2f}" if current_rsi is not None else "N/A")
        
        if not df.empty:
            fig = create_chart(df, f"Nifty {interval_display}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No chart data available - check Angel One credentials and connection")
    
    with col2:
        st.header("Options (Mock Data)")
        st.info("Note: This uses mock options data. Integrate with Angel One options chain API when available.")
        
        option_summary = create_mock_options_data(current_price)
        
        if option_summary is not None:
            st.info(f"Spot: ₹{current_price:.2f}" if current_price else "Price: N/A")
            st.dataframe(option_summary, use_container_width=True)
            
            if enable_signals and not df.empty and is_market_hours():
                check_signals(df, option_summary, current_price, proximity)
        else:
            st.error("Options data unavailable")
    
    # Status information
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time}")
    
    if api.authenticated:
        st.sidebar.success("✅ Angel One Connected")
    else:
        st.sidebar.error("❌ Angel One Not Connected")
    
    if st.sidebar.button("Test Telegram"):
        send_telegram("🔔 Test message from Nifty Analyzer (Angel One)")
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()
