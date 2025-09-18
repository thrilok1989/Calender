import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(page_title="Nifty Analyzer", page_icon="📈", layout="wide")

# Constants
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "")

def is_market_hours():
    """Check if it's market hours"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    if now.weekday() >= 5:  # Weekend
        return False
    
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return market_start <= now <= market_end

# Auto-refresh only during market hours
if is_market_hours():
    st_autorefresh(interval=35000, key="refresh")
else:
    st.info("Market is closed. Auto-refresh disabled.")

class DhanAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
    
    def get_intraday_data(self, interval="5", days_back=1):
        """Fetch intraday data"""
        url = "https://api.dhan.co/v2/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "securityId": str(NIFTY_SCRIP),
            "exchangeSegment": NIFTY_SEG,
            "instrument": "INDEX",
            "interval": interval,
            "oi": False,
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_ltp_data(self):
        """Get last traded price"""
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = {NIFTY_SEG: [NIFTY_SCRIP]}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except:
            return None

def get_option_chain(expiry):
    """Fetch option chain data"""
    url = "https://api.dhan.co/v2/optionchain"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_expiry_list():
    """Get available expiry dates"""
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def send_telegram(message):
    """Send message to Telegram with proper error handling"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.error("❌ Telegram credentials not configured in secrets")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Clean message to avoid HTML parsing issues
    # Replace problematic characters that cause HTML parsing errors
    clean_message = message.replace("<", "less than ").replace(">", "greater than ")
    clean_message = clean_message.replace("&", "and")
    
    # Try both string and integer chat ID formats
    chat_ids_to_try = [TELEGRAM_CHAT_ID]
    if TELEGRAM_CHAT_ID.isdigit() or (TELEGRAM_CHAT_ID.startswith('-') and TELEGRAM_CHAT_ID[1:].isdigit()):
        chat_ids_to_try.append(int(TELEGRAM_CHAT_ID))
    
    for chat_id in chat_ids_to_try:
        payload = {
            "chat_id": chat_id, 
            "text": clean_message, 
            "disable_web_page_preview": True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                st.success(f"✅ Telegram message sent successfully!")
                return True
            else:
                error_data = response.json()
                error_msg = error_data.get('description', 'Unknown error')
                st.error(f"❌ Telegram API Error: {error_msg}")
                
                # Common error solutions
                if "chat not found" in error_msg.lower():
                    st.info("💡 Solution: Make sure the bot has been started by sending /start to your bot first")
                elif "bot was blocked" in error_msg.lower():
                    st.info("💡 Solution: Unblock the bot in your Telegram and send /start")
                elif "unauthorized" in error_msg.lower():
                    st.info("💡 Solution: Check your bot token in secrets")
                
                return False
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return False
    
    return False

def process_candle_data(data):
    """Process raw candle data into DataFrame"""
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
    
    ist = pytz.timezone('Asia/Kolkata')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
    return df

def find_pivots(prices, length, pivot_type='high'):
    """Find pivot highs or lows"""
    if len(prices) < length * 2 + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    pivots = pd.Series(index=prices.index, dtype=float)
    
    for i in range(length, len(prices) - length):
        current = prices.iloc[i]
        left_side = prices.iloc[i-length:i]
        right_side = prices.iloc[i+1:i+length+1]
        
        if pivot_type == 'high':
            if current > left_side.max() and current > right_side.max():
                pivots.iloc[i] = current
        else:  # low
            if current < left_side.min() and current < right_side.min():
                pivots.iloc[i] = current
                
    return pivots

def get_pivots(df, timeframe="5", length=4):
    """Enhanced pivot detection"""
    if df.empty:
        return []
    
    rule_map = {"3": "3min", "5": "5min", "10": "10min", "15": "15min"}
    rule = rule_map.get(timeframe, "5min")
    
    df_temp = df.set_index('datetime')
    try:
        resampled = df_temp.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(resampled) < length * 2 + 1:
            return []
        
        pivot_highs = find_pivots(resampled['high'], length, 'high')
        pivot_lows = find_pivots(resampled['low'], length, 'low')
        
        pivots = []
        
        # Process highs
        valid_highs = pivot_highs.dropna()
        if len(valid_highs) > 1:
            for timestamp, value in valid_highs[:-1].items():
                pivots.append({
                    'type': 'high', 'timeframe': timeframe, 'timestamp': timestamp, 
                    'value': float(value), 'confirmed': True
                })
            
            if len(valid_highs) >= 1:
                last_high = valid_highs.iloc[-1]
                last_high_time = valid_highs.index[-1]
                pivots.append({
                    'type': 'high', 'timeframe': timeframe, 'timestamp': last_high_time,
                    'value': float(last_high), 'confirmed': False
                })
        
        # Process lows
        valid_lows = pivot_lows.dropna()
        if len(valid_lows) > 1:
            for timestamp, value in valid_lows[:-1].items():
                pivots.append({
                    'type': 'low', 'timeframe': timeframe, 'timestamp': timestamp,
                    'value': float(value), 'confirmed': True
                })
            
            if len(valid_lows) >= 1:
                last_low = valid_lows.iloc[-1] 
                last_low_time = valid_lows.index[-1]
                pivots.append({
                    'type': 'low', 'timeframe': timeframe, 'timestamp': last_low_time, 
                    'value': float(last_low), 'confirmed': False
                })
        
        return pivots
        
    except Exception as e:
        return []

def get_nearby_pivot_levels(df, current_price, proximity=5.0):
    """Get confirmed pivot levels near current price"""
    if df.empty:
        return []
        
    nearby_levels = []
    timeframes = ["5", "10", "15"]
    
    for timeframe in timeframes:
        pivots = get_pivots(df, timeframe, length=4)
        
        for pivot in pivots:
            if not pivot.get('confirmed', True):
                continue
                
            distance = abs(current_price - pivot['value'])
            if distance <= proximity:
                level_type = 'resistance' if pivot['type'] == 'high' else 'support'
                nearby_levels.append({
                    'type': level_type, 'pivot_type': pivot['type'], 'value': pivot['value'],
                    'timeframe': timeframe, 'distance': distance, 'timestamp': pivot['timestamp'],
                    'confirmed': pivot['confirmed']
                })
    
    nearby_levels.sort(key=lambda x: x['distance'])
    return nearby_levels

def calculate_rsi(df, periods=14):
    """Calculate RSI"""
    if len(df) < periods + 1:
        return pd.Series(index=df.index, dtype=float)
    
    price_changes = df['close'].diff()
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)
    
    rsi_values = pd.Series(index=df.index, dtype=float)
    
    initial_avg_gain = gains.rolling(window=periods).mean().iloc[periods-1]
    initial_avg_loss = losses.rolling(window=periods).mean().iloc[periods-1]
    
    if initial_avg_loss == 0:
        rsi_values.iloc[periods] = 100
    else:
        rs = initial_avg_gain / initial_avg_loss
        rsi_values.iloc[periods] = 100 - (100 / (1 + rs))
    
    alpha = 1.0 / periods
    avg_gain = initial_avg_gain
    avg_loss = initial_avg_loss
    
    for i in range(periods + 1, len(df)):
        avg_gain = alpha * gains.iloc[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses.iloc[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            rsi_values.iloc[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values.iloc[i] = 100 - (100 / (1 + rs))
    
    return rsi_values

def create_chart(df, title):
    """Create enhanced chart with pivots and RSI"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2], subplot_titles=("Price Chart", "Volume", "RSI (14)")
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Volume
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)
    
    # RSI
    if len(df) > 14:
        rsi_values = calculate_rsi(df)
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=rsi_values, name='RSI',
            line=dict(color='#ffaa00', width=2), mode='lines'
        ), row=3, col=1)
        
        # RSI levels
        for level, color, label in [(70, "red", "Overbought"), (30, "green", "Oversold"), (50, "gray", "Midline")]:
            fig.add_hline(y=level, line_dash="dash" if level != 50 else "dot", 
                         line_color=color, annotation_text=f"{label} ({level})", row=3, col=1)
        
        fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    # Add pivot levels
    if len(df) > 50:
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", '#4444ff']
        x_start, x_end = df['datetime'].min(), df['datetime'].max()
        
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            recent_highs = [p for p in pivots if p['type'] == 'high'][-5:]
            recent_lows = [p for p in pivots if p['type'] == 'low'][-5:]
            
            for pivot_list, label_prefix in [(recent_highs, "H"), (recent_lows, "L")]:
                for pivot in pivot_list:
                    line_style = "solid" if pivot.get('confirmed', True) else "dash"
                    line_width = 2 if pivot.get('confirmed', True) else 1
                    
                    # Main line
                    fig.add_shape(
                        type="line", x0=x_start, x1=x_end, y0=pivot['value'], y1=pivot['value'],
                        line=dict(color=color, width=line_width, dash=line_style), row=1, col=1
                    )
                    
                    # Shadow
                    fig.add_shape(
                        type="line", x0=x_start, x1=x_end, y0=pivot['value'], y1=pivot['value'],
                        line=dict(color=color, width=5),
                        opacity=0.15 if pivot.get('confirmed', True) else 0.08, row=1, col=1
                    )
                    
                    # Label
                    status = "✓" if pivot.get('confirmed', True) else "?"
                    fig.add_annotation(
                        x=x_end, y=pivot['value'], text=f"{tf}M {label_prefix} {status}: {pivot['value']:.1f}",
                        showarrow=False, xshift=20, font=dict(size=9, color=color), row=1, col=1
                    )
    
    fig.update_layout(
        title=title, template='plotly_dark', height=700,
        xaxis_rangeslider_visible=False, showlegend=False
    )
    
    # Update x-axis labels
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=3, col=1)
    
    return fig

# News Analysis Functions
def fetch_alpha_vantage_news():
    """Fetch news with rate limiting"""
    if not ALPHA_VANTAGE_KEY:
        return [{'title': 'Alpha Vantage API key not configured', 'summary': 'Please add ALPHA_VANTAGE_KEY to secrets',
                'source': 'System', 'published_at': datetime.now().isoformat(), 'sentiment_score': 0}]
    
    try:
        current_time = time.time()
        if 'last_news_request' in st.session_state:
            time_since_last = current_time - st.session_state.last_news_request
            if time_since_last < 900:  # 15 minutes
                if 'alpha_vantage_news' in st.session_state:
                    return st.session_state.alpha_vantage_news
                return []
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=NIFTY&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 429:
            if 'alpha_vantage_news' in st.session_state:
                return st.session_state.alpha_vantage_news
            return []
        
        response.raise_for_status()
        data = response.json()
        feed = data.get('feed', [])[:5]
        
        processed_news = []
        for item in feed:
            sentiment_info = item.get('overall_sentiment', {})
            sentiment_score = sentiment_info.get('score', 0) if isinstance(sentiment_info, dict) else 0
            
            processed_news.append({
                'title': item.get('title', ''), 'summary': item.get('summary', '')[:200],
                'url': item.get('url', ''), 'source': item.get('source', 'Unknown'),
                'published_at': item.get('time_published', ''), 'sentiment_score': sentiment_score
            })
        
        st.session_state.last_news_request = current_time
        st.session_state.alpha_vantage_news = processed_news
        return processed_news
        
    except:
        return st.session_state.get('alpha_vantage_news', [])

def analyze_news_sentiment(news_items):
    """Analyze sentiment from news"""
    if not news_items or any('error' in item.get('title', '').lower() for item in news_items):
        return {"overall": "neutral", "score": 0, "bullish_count": 0, "bearish_count": 0}
    
    sentiment_scores = []
    for item in news_items:
        sentiment_score = item.get('sentiment_score', 0)
        if 0 <= sentiment_score <= 1:
            sentiment_scores.append(sentiment_score * 2 - 1)
        else:
            sentiment_scores.append(sentiment_score)
    
    if sentiment_scores:
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        
        if avg_score > 0.2:
            overall = "bullish"
        elif avg_score < -0.2:
            overall = "bearish"
        else:
            overall = "neutral"
            
        return {
            "overall": overall, "score": round(avg_score, 3),
            "bullish_count": sum(1 for score in sentiment_scores if score > 0.2),
            "bearish_count": sum(1 for score in sentiment_scores if score < -0.2),
            "neutral_count": sum(1 for score in sentiment_scores if -0.2 <= score <= 0.2),
            "confidence": min(abs(avg_score) * 2, 1.0)
        }
    
    return {"overall": "neutral", "score": 0, "bullish_count": 0, "bearish_count": 0}

# Technical Analysis Functions
def get_market_trend(df, period=20):
    """Market trend analysis"""
    if len(df) < period:
        return "neutral"
    
    sma_short = df['close'].rolling(10).mean().iloc[-1]
    sma_long = df['close'].rolling(20).mean().iloc[-1]
    
    if sma_short > sma_long * 1.002:
        return "bullish"
    elif sma_short < sma_long * 0.998:
        return "bearish"
    else:
        return "neutral"

def check_pcr_condition(pcr, signal_type):
    """Check PCR condition for signals"""
    if signal_type == "bullish":
        return pcr > 1.3  # More puts than calls = bullish
    elif signal_type == "bearish":
        return pcr < 0.7  # More calls than puts = bearish
    return False

def get_pcr_condition_text(pcr, signal_type):
    """Get PCR condition text without problematic symbols"""
    if signal_type == "bullish":
        threshold = 1.3
        condition = "above" if pcr > threshold else "below"
        return f"PCR {pcr:.2f} ({condition} {threshold} for bull)"
    elif signal_type == "bearish":
        threshold = 0.7
        condition = "below" if pcr < threshold else "above"
        return f"PCR {pcr:.2f} ({condition} {threshold} for bear)"
    return f"PCR {pcr:.2f}"

def get_momentum_score(df, periods=14):
    """Calculate momentum score from RSI"""
    if len(df) < periods + 1:
        return 5
    
    rsi_values = calculate_rsi(df, periods)
    latest_rsi = rsi_values.iloc[-1]
    return max(1, min(10, int(latest_rsi / 10))) if not pd.isna(latest_rsi) else 5

def get_market_volatility(df, period=14):
    """Market volatility classification"""
    if len(df) < period:
        return "normal"
    
    returns = df['close'].pct_change().dropna()
    volatility = returns.rolling(period).std().iloc[-1] * 100
    
    if volatility > 2.0:
        return "high"
    elif volatility < 0.5:
        return "low"
    else:
        return "normal"

def is_good_signal_time():
    """Time-based filter"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    market_start = now.replace(hour=9, minute=15, second=0)
    market_end = now.replace(hour=15, minute=15, second=0)
    return market_start <= now <= market_end

def get_price_momentum_profile(df, periods=20):
    """Price momentum analysis (replaces volume profile)"""
    if len(df) < periods:
        return {"profile": "insufficient_data", "strength": 1}
    
    # Use price momentum instead of volume
    returns = df['close'].pct_change().tail(periods)
    recent_momentum = abs(returns.tail(5).mean())
    avg_momentum = abs(returns.mean())
    momentum_ratio = recent_momentum / avg_momentum if avg_momentum > 0 else 1
    
    if momentum_ratio > 2.0:
        profile, strength = "explosive", 2.5
    elif momentum_ratio > 1.5:
        profile, strength = "strong", 2.0
    elif momentum_ratio > 1.2:
        profile, strength = "above_average", 1.5
    elif momentum_ratio > 0.8:
        profile, strength = "normal", 1.0
    else:
        profile, strength = "weak", 0.7
    
    return {"profile": profile, "strength": round(strength, 2), "momentum_ratio": momentum_ratio}

def detect_market_regime(df, short_period=10, long_period=30):
    """Market regime detection"""
    if len(df) < long_period:
        return "unknown"
    
    short_ma = df['close'].rolling(short_period).mean().iloc[-1]
    long_ma = df['close'].rolling(long_period).mean().iloc[-1]
    
    trend_direction = "bullish" if short_ma > long_ma else "bearish"
    
    # Simple trend strength using price correlation
    try:
        x_values = range(len(df.tail(long_period)))
        y_values = df['close'].tail(long_period).values
        correlation = np.corrcoef(x_values, y_values)[0, 1]
        r_squared = correlation ** 2
        
        if r_squared > 0.3:
            return f"{trend_direction}_trending"
        elif r_squared > 0.1:
            return f"weak_{trend_direction}_trend"
        else:
            return "ranging"
    except:
        return "ranging"

def check_breakout_pattern(df, current_price, lookback=20):
    """Breakout detection (without volume dependency)"""
    if len(df) < lookback:
        return False, "insufficient_data"
    
    recent_data = df.tail(lookback)
    recent_high = recent_data['high'].max()
    recent_low = recent_data['low'].min()
    
    # ATR calculation
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    
    breakout_threshold = atr * 0.5
    
    # Price momentum confirmation instead of volume
    recent_returns = df['close'].pct_change().tail(5)
    momentum_strength = abs(recent_returns.mean()) > abs(df['close'].pct_change().tail(20).mean()) * 1.5
    
    if current_price > recent_high + breakout_threshold and momentum_strength:
        return True, "upside_breakout"
    elif current_price < recent_low - breakout_threshold and momentum_strength:
        return True, "downside_breakout"
    elif current_price > recent_high + (breakout_threshold * 0.5):
        return True, "weak_upside_breakout"
    elif current_price < recent_low - (breakout_threshold * 0.5):
        return True, "weak_downside_breakout"
    
    return False, "range_bound"

def get_session_performance():
    """Get current session"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    if 9 <= now.hour < 11:
        return "opening_session"
    elif 11 <= now.hour < 13:
        return "mid_session" 
    elif 13 <= now.hour < 15:
        return "afternoon_session"
    else:
        return "closing_session"

def analyze_options(expiry):
    """Analyze options data"""
    option_data = get_option_chain(expiry)
    if not option_data or 'data' not in option_data:
        return None, None
    
    data = option_data['data']
    underlying = data['last_price']
    oc_data = data['oc']
    
    calls, puts = [], []
    for strike, strike_data in oc_data.items():
        if 'ce' in strike_data:
            ce_data = strike_data['ce']
            ce_data['strikePrice'] = float(strike)
            calls.append(ce_data)
        if 'pe' in strike_data:
            pe_data = strike_data['pe']
            pe_data['strikePrice'] = float(strike)
            puts.append(pe_data)
    
    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
    
    # Rename columns
    rename_map = {
        'last_price': 'lastPrice', 'oi': 'openInterest', 'previous_oi': 'previousOpenInterest',
        'top_ask_quantity': 'askQty', 'top_bid_quantity': 'bidQty', 'volume': 'totalTradedVolume'
    }
    for old, new in rename_map.items():
        df.rename(columns={f"{old}_CE": f"{new}_CE", f"{old}_PE": f"{new}_PE"}, inplace=True)
    
    df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
    df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
    
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df_filtered = df[abs(df['strikePrice'] - atm_strike) <= 100]
    
    df_filtered['Zone'] = df_filtered['strikePrice'].apply(
        lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM'
    )
    
    # Calculate biases
    bias_results = []
    for _, row in df_filtered.iterrows():
        chg_oi_bias = "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish"
        volume_bias = "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish"
        
        ask_ce, ask_pe = row.get('askQty_CE', 0), row.get('askQty_PE', 0)
        bid_ce, bid_pe = row.get('bidQty_CE', 0), row.get('bidQty_PE', 0)
        
        ask_bias = "Bearish" if ask_ce > ask_pe else "Bullish"
        bid_bias = "Bullish" if bid_ce > bid_pe else "Bearish"
        
        ce_oi, pe_oi = row['openInterest_CE'], row['openInterest_PE']
        level = "Support" if pe_oi > 1.12 * ce_oi else "Resistance" if ce_oi > 1.12 * pe_oi else "Neutral"
        
        bias_results.append({
            "Strike": row['strikePrice'], "Zone": row['Zone'], "Level": level,
            "ChgOI_Bias": chg_oi_bias, "Volume_Bias": volume_bias,
            "Ask_Bias": ask_bias, "Bid_Bias": bid_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else 0, 2),
            "changeinOpenInterest_CE": row['changeinOpenInterest_CE'],
            "changeinOpenInterest_PE": row['changeinOpenInterest_PE']
        })
    
    return underlying, pd.DataFrame(bias_results)

def check_signals(df, option_data, current_price, proximity=5):
    """Comprehensive signal checking with 8 signals"""
    if df.empty or option_data is None or not current_price:
        return
    
    # Get news and market analysis
    news_items = fetch_alpha_vantage_news()
    news_sentiment = analyze_news_sentiment(news_items)
    market_trend = get_market_trend(df)
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return
    
    row = atm_data.iloc[0]
    
    # Get current RSI
    rsi_values = calculate_rsi(df)
    current_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 and not pd.isna(rsi_values.iloc[-1]) else 50
    
    # Bias checks
    bias_aligned_bullish = all([
        row['ChgOI_Bias'] == 'Bullish', row['Volume_Bias'] == 'Bullish',
        row['Ask_Bias'] == 'Bullish', row['Bid_Bias'] == 'Bullish'
    ])
    
    bias_aligned_bearish = all([
        row['ChgOI_Bias'] == 'Bearish', row['Volume_Bias'] == 'Bearish',
        row['Ask_Bias'] == 'Bearish', row['Bid_Bias'] == 'Bearish'
    ])
    
    # PCR conditions
    pcr = row['PCR']
    pcr_bullish = check_pcr_condition(pcr, "bullish")
    pcr_bearish = check_pcr_condition(pcr, "bearish")
    
    # News alignment
    news_emoji = "📈" if news_sentiment["overall"] == "bullish" else "📉" if news_sentiment["overall"] == "bearish" else "📊"
    
    # Get comprehensive info for messages
    comprehensive_bias_info = get_comprehensive_bias_info(df, option_data, current_price, news_sentiment, market_trend, current_rsi)
    
    # Signal 1: Primary Signal - Pivot + Bias alignment + PCR
    nearby_levels = get_nearby_pivot_levels(df, current_price, proximity)
    if nearby_levels:
        pivot_level = nearby_levels[0]
        primary_bullish = row['Level'] == 'Support' and bias_aligned_bullish and pivot_level['type'] == 'support' and pcr_bullish
        primary_bearish = row['Level'] == 'Resistance' and bias_aligned_bearish and pivot_level['type'] == 'resistance' and pcr_bearish
        
        if primary_bullish or primary_bearish:
            signal_type = "CALL" if primary_bullish else "PUT"
            price_diff = current_price - pivot_level['value']
            pcr_text = get_pcr_condition_text(pcr, "bullish" if primary_bullish else "bearish")
            
            message = f"""
🚨 PRIMARY NIFTY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M {pivot_level['type'].title()} at ₹{pivot_level['value']:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.1f}
📈 {pcr_text} {'✅' if pcr_bullish or pcr_bearish else '❌'}

{news_emoji} News Sentiment: {news_sentiment['overall'].title()}
Conditions: {row['Level']}, All Bias Aligned, Confirmed Pivot, PCR Condition Met

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
            send_telegram(message)
            st.success(f"PRIMARY {signal_type} signal sent!")
    
    # Signal 2: Secondary Signal - OI Dominance + PCR
    ce_chg_oi = abs(row.get('changeinOpenInterest_CE', 0))
    pe_chg_oi = abs(row.get('changeinOpenInterest_PE', 0))
    
    put_dominance = pe_chg_oi > 1.3 * ce_chg_oi if ce_chg_oi > 0 else False
    call_dominance = ce_chg_oi > 1.3 * pe_chg_oi if pe_chg_oi > 0 else False
    
    secondary_bullish = bias_aligned_bullish and put_dominance and pcr_bullish
    secondary_bearish = bias_aligned_bearish and call_dominance and pcr_bearish
    
    if secondary_bullish or secondary_bearish:
        signal_type = "CALL" if secondary_bullish else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if secondary_bullish and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        pcr_text = get_pcr_condition_text(pcr, "bullish" if secondary_bullish else "bearish")
        
        message = f"""
⚡ SECONDARY NIFTY {signal_type} SIGNAL - OI DOMINANCE ⚡

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.1f}
📈 {pcr_text} {'✅' if pcr_bullish or pcr_bearish else '❌'}

{news_emoji} News Sentiment: {news_sentiment['overall'].title()}
🔥 OI Dominance: {'PUT' if secondary_bullish else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
📊 All Bias Aligned + PCR Condition

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"SECONDARY {signal_type} signal sent!")

    # Signal 3: Breakout + Momentum + PCR
    is_breakout, breakout_type = check_breakout_pattern(df, current_price)
    momentum_score = get_momentum_score(df)
    price_profile = get_price_momentum_profile(df)
    
    breakout_bullish_signal = (
        is_breakout and 
        breakout_type == "upside_breakout" and
        momentum_score >= 6 and
        price_profile["strength"] >= 1.2 and
        bias_aligned_bullish and
        pcr_bullish
    )
    
    breakout_bearish_signal = (
        is_breakout and 
        breakout_type == "downside_breakout" and
        momentum_score <= 4 and
        price_profile["strength"] >= 1.2 and
        bias_aligned_bearish and
        pcr_bearish
    )
    
    if breakout_bullish_signal or breakout_bearish_signal:
        signal_type = "CALL" if breakout_bullish_signal else "PUT"
        
        message = f"""
💥 THIRD SIGNAL - BREAKOUT + MOMENTUM {signal_type} 💥

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.1f}
📈 PCR: {pcr:.2f} ({'✅' if pcr_bullish or pcr_bearish else '❌'})

{news_emoji} News Sentiment: {news_sentiment['overall'].title()}
🚀 Breakout Type: {breakout_type.replace('_', ' ').title()}
📊 Momentum Score: {momentum_score}/10
🔊 Price Profile: {price_profile["profile"].title()} ({price_profile["strength"]:.1f}x)

All ATM Biases + Breakout + PCR Confirmed

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"THIRD BREAKOUT {signal_type} signal sent!")

    # Signal 4: All Bias Aligned + PCR
    if (bias_aligned_bullish and pcr_bullish) or (bias_aligned_bearish and pcr_bearish):
        signal_type = "CALL" if bias_aligned_bullish else "PUT"
        
        message = f"""
🎯 FOURTH SIGNAL - ALL BIAS ALIGNED + PCR {signal_type} 🎯

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.1f}
📈 PCR: {pcr:.2f} ({'✅' if pcr_bullish or pcr_bearish else '❌'})

{news_emoji} News: {news_sentiment['overall'].title()}
All ATM Biases Aligned + PCR Condition: {signal_type}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"FOURTH {signal_type} signal sent!")

    # Signal 5: Enhanced Primary (No volume dependency)
    if nearby_levels:
        pivot_level = nearby_levels[0]
        market_regime = detect_market_regime(df)
        session = get_session_performance()
        time_ok = is_good_signal_time()
        volatility = get_market_volatility(df)
        
        enhanced_bullish = (
            bias_aligned_bullish and
            row['Level'] == 'Support' and
            pivot_level['type'] == 'support' and
            pcr_bullish and
            market_trend in ["bullish", "neutral"] and
            volatility == "normal" and
            time_ok and
            momentum_score >= 6 and
            price_profile["strength"] >= 1.2
        )
        
        enhanced_bearish = (
            bias_aligned_bearish and
            row['Level'] == 'Resistance' and
            pivot_level['type'] == 'resistance' and
            pcr_bearish and
            market_trend in ["bearish", "neutral"] and
            volatility == "normal" and
            time_ok and
            momentum_score <= 4 and
            price_profile["strength"] >= 1.2
        )
        
        if enhanced_bullish or enhanced_bearish:
            signal_type = "CALL" if enhanced_bullish else "PUT"
            price_diff = current_price - pivot_level['value']
            
            message = f"""
🌟 FIFTH SIGNAL - ENHANCED PRIMARY {signal_type} 🌟

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M {pivot_level['type'].title()} at ₹{pivot_level['value']:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.1f}
📈 PCR: {pcr:.2f} ({'✅' if pcr_bullish or pcr_bearish else '❌'})

📊 Market Regime: {market_regime.replace('_', ' ').title()}
🔊 Price Profile: {price_profile["profile"].title()} ({price_profile["strength"]:.1f}x)
🚀 Momentum: {momentum_score}/10
⏰ Session: {session.replace('_', ' ').title()}
{news_emoji} News: {news_sentiment['overall'].title()}

All Premium Conditions Met + PCR

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
            send_telegram(message)
            st.success(f"ENHANCED FIFTH {signal_type} signal sent!")

    # Signal 6: Confluence + Flow + PCR
    strong_confluence_bullish = (
        bias_aligned_bullish and
        put_dominance and
        pcr_bullish and
        market_trend in ["bullish", "neutral"] and
        volatility == "normal" and
        time_ok and
        momentum_score >= 6 and
        price_profile["strength"] >= 1.3
    )
    
    strong_confluence_bearish = (
        bias_aligned_bearish and
        call_dominance and
        pcr_bearish and
        market_trend in ["bearish", "neutral"] and
        volatility == "normal" and
        time_ok and
        momentum_score <= 4 and
        price_profile["strength"] >= 1.3
    )
    
    if strong_confluence_bullish or strong_confluence_bearish:
        signal_type = "CALL" if strong_confluence_bullish else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if strong_confluence_bullish and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        message = f"""
🚀 SIXTH SIGNAL - CONFLUENCE + FLOW + PCR {signal_type} 🚀

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.1f}
📈 PCR: {pcr:.2f} ({'✅' if pcr_bullish or pcr_bearish else '❌'})

🔥 OI Dominance: {'PUT' if strong_confluence_bullish else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
🔊 Price Profile: {price_profile["profile"].title()} ({price_profile["strength"]:.1f}x)
🚀 Momentum: {momentum_score}/10
{news_emoji} News: {news_sentiment['overall'].title()}

All Premium Filters + PCR Passed

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"CONFLUENCE SIXTH {signal_type} signal sent!")

    # Signal 7: MAJOR SIGNAL - Triple Confirmation + Strong PCR
    rsi_extreme_bullish = current_rsi < 35  # Oversold
    rsi_extreme_bearish = current_rsi > 65  # Overbought
    
    major_bullish = (
        bias_aligned_bullish and
        row['Level'] == 'Support' and
        pcr > 1.5 and  # Very strong PCR for major signal
        put_dominance and
        (rsi_extreme_bullish or momentum_score >= 7) and
        market_trend != "bearish"
    )
    
    major_bearish = (
        bias_aligned_bearish and
        row['Level'] == 'Resistance' and
        pcr < 0.6 and  # Very strong PCR for major signal
        call_dominance and
        (rsi_extreme_bearish or momentum_score <= 3) and
        market_trend != "bullish"
    )
    
    if major_bullish or major_bearish:
        signal_type = "CALL" if major_bullish else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if major_bullish and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        # Create safe PCR text for major signals
        if major_bullish:
            pcr_strength = "VERY STRONG" if pcr > 1.5 else "STRONG"
            pcr_description = f"PCR {pcr:.2f} (above 1.5 threshold)"
        else:
            pcr_strength = "VERY STRONG" if pcr < 0.6 else "STRONG" 
            pcr_description = f"PCR {pcr:.2f} (below 0.6 threshold)"
        
        message = f"""
MAJOR SIGNAL - SEVENTH {signal_type}

FULL ALERT - STRONG CONVICTION SIGNAL

Spot: ₹{current_price:.2f}
ATM: {row['Strike']}
RSI: {current_rsi:.1f} {'(OVERSOLD)' if rsi_extreme_bullish else '(OVERBOUGHT)' if rsi_extreme_bearish else ''}
{pcr_description} {pcr_strength}

Level: {row['Level']}
OI Dominance: {'PUT' if major_bullish else 'CALL'} ChgOI {dominance_ratio:.1f}x
Momentum: {momentum_score}/10
Trend: {market_trend.title()}
News: {news_sentiment['overall'].title()}

TRIPLE CONFIRMATION:
✅ All Biases Aligned
✅ Strong PCR Condition (threshold met)
✅ OI Dominance + Level Support

⚠️ HIGH CONVICTION TRADE SETUP ⚠️

Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"MAJOR SEVENTH {signal_type} SIGNAL SENT!")

    # Signal 8: Ultimate Signal - All Conditions + Extreme PCR + RSI
    ultimate_bullish = (
        bias_aligned_bullish and
        pcr > 1.4 and
        put_dominance and
        row['Level'] == 'Support' and
        current_rsi < 40 and
        momentum_score >= 6 and
        price_profile["strength"] >= 1.5 and
        nearby_levels and
        nearby_levels[0]['type'] == 'support'
    )
    
    ultimate_bearish = (
        bias_aligned_bearish and
        pcr < 0.65 and
        call_dominance and
        row['Level'] == 'Resistance' and
        current_rsi > 60 and
        momentum_score <= 4 and
        price_profile["strength"] >= 1.5 and
        nearby_levels and
        nearby_levels[0]['type'] == 'resistance'
    )
    
    if ultimate_bullish or ultimate_bearish:
        signal_type = "CALL" if ultimate_bullish else "PUT"
        pivot_level = nearby_levels[0] if nearby_levels else None
        price_diff = current_price - pivot_level['value'] if pivot_level else 0
        dominance_ratio = pe_chg_oi / ce_chg_oi if ultimate_bullish and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        # Create safe PCR text for ultimate signals
        if ultimate_bullish:
            pcr_description = f"PCR {pcr:.2f} (above 1.4 extreme threshold)"
        else:
            pcr_description = f"PCR {pcr:.2f} (below 0.65 extreme threshold)"
        
        message = f"""
ULTIMATE SIGNAL - EIGHTH {signal_type}

MAXIMUM CONVICTION - RARE SETUP

Spot: ₹{current_price:.2f}
Pivot: {pivot_level['timeframe']}M {pivot_level['type'].title()} at ₹{pivot_level['value']:.2f} ({price_diff:+.2f})
ATM: {row['Strike']}
RSI: {current_rsi:.1f} {'EXTREME OVERSOLD' if ultimate_bullish else 'EXTREME OVERBOUGHT'}
{pcr_description} EXTREME {'BULLISH' if ultimate_bullish else 'BEARISH'}

PERFECT ALIGNMENT:
✅ All ATM Biases Aligned
✅ Extreme PCR Condition (threshold met)
✅ RSI Extreme ({'below 40' if ultimate_bullish else 'above 60'})
✅ OI Dominance {dominance_ratio:.1f}x
✅ Level + Pivot Confluence
✅ Strong Price Momentum ({price_profile["strength"]:.1f}x)

PRIMARY: All Bias + Level + Pivot
SECONDARY: Extreme PCR + OI Dominance
BONUS: RSI Extreme + Momentum

ULTRA HIGH CONVICTION
POSITION SIZE: MAXIMUM
RISK/REWARD: EXCELLENT

Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_bias_info}
"""
        send_telegram(message)
        st.success(f"ULTIMATE EIGHTH {signal_type} SIGNAL SENT!")

def get_comprehensive_bias_info(df, option_data, current_price, news_sentiment, market_trend, current_rsi):
    """Get comprehensive bias information with updated quantitative data"""
    try:
        atm_data = option_data[option_data['Zone'] == 'ATM']
        if atm_data.empty:
            return "Options data unavailable for bias analysis"
        
        row = atm_data.iloc[0]
        
        # Technical metrics (removed volume dependencies)
        volatility = get_market_volatility(df)
        momentum_score = get_momentum_score(df)
        price_profile = get_price_momentum_profile(df)
        market_regime = detect_market_regime(df)
        session = get_session_performance()
        time_ok = is_good_signal_time()
        
        bias_info = f"""
📊 COMPREHENSIVE BIAS ANALYSIS 📊

🔹 OPTIONS BIASES:
• ChgOI Bias: {row['ChgOI_Bias']}
• Volume Bias: {row['Volume_Bias']} 
• Ask Bias: {row['Ask_Bias']}
• Bid Bias: {row['Bid_Bias']}
• Level Bias: {row['Level']} (OI-based)
• PCR: {row['PCR']} (PE/CE ratio)

🔹 TECHNICAL BIASES:
• Market Trend: {market_trend.title()}
• RSI: {current_rsi:.1f} ({'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'})
• Volatility: {volatility.title()}
• Momentum: {momentum_score}/10
• Price Profile: {price_profile["profile"].title()} ({price_profile["strength"]:.1f}x)
• Market Regime: {market_regime.replace('_', ' ').title()}
• Session: {session.replace('_', ' ').title()}
• Timing: {'Good' if time_ok else 'Poor'}

🔹 SENTIMENT BIASES:
• News Sentiment: {news_sentiment['overall'].title()} (Score: {news_sentiment['score']:.2f})

🔹 QUANTITATIVE DATA:
• CE ChgOI: {row.get('changeinOpenInterest_CE', 0):,}
• PE ChgOI: {row.get('changeinOpenInterest_PE', 0):,}
• CE OI: {row.get('openInterest_CE', 0):,}
• PE OI: {row.get('openInterest_PE', 0):,}
• CE Volume: {row.get('totalTradedVolume_CE', 0):,}
• PE Volume: {row.get('totalTradedVolume_PE', 0):,}
• CE Ask: {row.get('askQty_CE', 0):,}
• CE Bid: {row.get('bidQty_CE', 0):,}
• PE Ask: {row.get('askQty_PE', 0):,}
• PE Bid: {row.get('bidQty_PE', 0):,}
• ATM Strike: {row['Strike']}
"""
        return bias_info
        
    except Exception as e:
        return f"Error calculating comprehensive bias: {str(e)}"

def main():
    st.title("📈 Nifty Trading Analyzer")
    
    # Show market status
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"⚠️ Market is closed. Current time: {current_time.strftime('%H:%M:%S IST')}")
        st.info("Market hours: Monday-Friday, 9:00 AM to 3:45 PM IST")
    
    # Sidebar settings with Telegram debugging
    st.sidebar.header("🎛️ Settings")
    interval = st.sidebar.selectbox("Timeframe", ["1", "3", "5", "10", "15"], index=2)
    enable_signals = st.sidebar.checkbox("Enable Signals", value=True)
    proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
    
    # Telegram Configuration Check
    st.sidebar.subheader("📱 Telegram Setup")
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.sidebar.success("✅ Telegram credentials configured")
        
        # Show configuration details (masked)
        if len(TELEGRAM_BOT_TOKEN) > 10:
            masked_token = TELEGRAM_BOT_TOKEN[:8] + "..." + TELEGRAM_BOT_TOKEN[-4:]
            st.sidebar.text(f"Bot Token: {masked_token}")
        st.sidebar.text(f"Chat ID: {TELEGRAM_CHAT_ID}")
        
        # Test button
        if st.sidebar.button("📤 Test Telegram"):
            test_message = f"""
🔔 Enhanced Test Message from Nifty Analyzer

📊 System Status: Online
🕒 Time: {current_time}
⚙️ All Systems Operational

🚀 Enhanced Features Active:
✅ 8 Advanced Signal Types
✅ PCR Integration (>1.3 Bull, <0.7 Bear)
✅ RSI Values in All Messages
✅ Price Momentum Analysis
✅ News Sentiment Display
✅ Major Signal with Full Alert
✅ Ultimate Signal (Max Conviction)
✅ Enhanced Quantitative Data
✅ Non-Volume Dependent Analysis
✅ Comprehensive Bias Analysis

🔥 New Signals:
• 7th: MAJOR SIGNAL (Triple Confirmation)
• 8th: ULTIMATE SIGNAL (Maximum Conviction)

📈 All systems optimized and ready!
"""
            success = send_telegram(test_message)
            if not success:
                st.sidebar.error("❌ Test message failed!")
        
        # Quick troubleshooting guide
        with st.sidebar.expander("🔧 Troubleshooting", expanded=False):
            st.write("""
**If messages aren't working:**

1. **Start your bot**: Send `/start` to your bot in Telegram
2. **Check Chat ID**: Use @userinfobot to get correct chat ID
3. **Verify Bot Token**: Make sure it's correct in secrets
4. **Bot permissions**: Ensure bot can send messages
5. **Group chats**: Use negative chat ID (-1234567890)

**How to get Chat ID:**
- Send message to @userinfobot in Telegram
- It will reply with your chat ID
- Use this exact number in secrets
""")
    else:
        st.sidebar.error("❌ Telegram not configured")
        with st.sidebar.expander("📝 Setup Instructions", expanded=True):
            st.write("""
**Step 1: Create Bot**
1. Message @BotFather on Telegram
2. Send `/newbot`
3. Follow instructions
4. Copy bot token

**Step 2: Get Chat ID**
1. Send message to @userinfobot
2. Copy your chat ID

**Step 3: Add to Secrets**
```
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

**Step 4: Start Bot**
Send `/start` to your bot
""")
    
    if st.sidebar.button("Refresh News"):
        for key in ['news_cache', 'last_news_request', 'alpha_vantage_news']:
            if key in st.session_state:
                del st.session_state[key]
        st.sidebar.success("News cache cleared!")
    
    api = DhanAPI()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chart")
        
        # Get data
        data = api.get_intraday_data(interval)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        # Get current price
        ltp_data = api.get_ltp_data()
        current_price = None
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        # Show metrics
        if not df.empty and len(df) > 1:
            prev_close = df['close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            col1_m, col2_m, col3_m = st.columns(3)
            with col1_m:
                st.metric("Price", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            with col2_m:
                st.metric("High", f"₹{df['high'].max():,.2f}")
            with col3_m:
                st.metric("Low", f"₹{df['low'].min():,.2f}")
        
        # Chart
        if not df.empty:
            fig = create_chart(df, f"Nifty {interval}min")
            st.plotly_chart(fig, use_container_width=True)
            
            # News analysis
            if current_price:
                news_items = fetch_alpha_vantage_news()
                news_sentiment = analyze_news_sentiment(news_items)
                
                if news_items:
                    st.subheader("Market News")
                    col_news1, col_news2, col_news3 = st.columns(3)
                    with col_news1:
                        st.metric("News Sentiment", news_sentiment["overall"].title())
                    with col_news2:
                        st.metric("Bullish Items", news_sentiment["bullish_count"])
                    with col_news3:
                        st.metric("Bearish Items", news_sentiment["bearish_count"])
                
                # Technical analysis (updated without volume dependency)
                if len(df) > 20:
                    market_trend = get_market_trend(df)
                    volatility = get_market_volatility(df)
                    momentum_score = get_momentum_score(df)
                    price_profile = get_price_momentum_profile(df)
                    market_regime = detect_market_regime(df)
                    time_ok = is_good_signal_time()
                    is_breakout, breakout_type = check_breakout_pattern(df, current_price)
                    session = get_session_performance()
                    
                    # Get current RSI
                    rsi_values = calculate_rsi(df)
                    current_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 and not pd.isna(rsi_values.iloc[-1]) else 50
                    
                    st.subheader("📊 Technical Analysis")
                    
                    # Primary metrics row
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        trend_color = "🟢" if market_trend == "bullish" else "🔴" if market_trend == "bearish" else "🟡"
                        st.metric("Trend", f"{trend_color} {market_trend.title()}")
                    with col_b:
                        rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                        st.metric("RSI", f"{rsi_color} {current_rsi:.1f}")
                    with col_c:
                        vol_color = "🟢" if volatility == "normal" else "🟡" if volatility == "low" else "🔴"
                        st.metric("Volatility", f"{vol_color} {volatility.title()}")
                    with col_d:
                        time_color = "🟢" if time_ok else "🔴"
                        st.metric("Timing", f"{time_color} {'Good' if time_ok else 'Poor'}")
                    
                    # Secondary metrics row  
                    col_e, col_f, col_g, col_h = st.columns(4)
                    with col_e:
                        momentum_color = "🟢" if momentum_score >= 6 else "🔴" if momentum_score <= 4 else "🟡"
                        st.metric("Momentum", f"{momentum_color} {momentum_score}/10")
                    with col_f:
                        regime_color = "🟢" if "trending" in market_regime else "🟡"
                        st.metric("Regime", f"{regime_color} {market_regime.replace('_', ' ').title()}")
                    with col_g:
                        price_prof_color = "🟢" if price_profile["strength"] >= 1.2 else "🟡" if price_profile["strength"] >= 1.0 else "🔴"
                        st.metric("Price Profile", f"{price_prof_color} {price_profile['profile'].title()}")
                    with col_h:
                        session_color = "🟢" if session in ["mid_session", "afternoon_session"] else "🟡"
                        st.metric("Session", f"{session_color} {session.replace('_', ' ').title()}")
                    
                    # Breakout status
                    if is_breakout:
                        breakout_color = "🚀" if breakout_type == "upside_breakout" else "⬇️"
                        st.info(f"{breakout_color} **Breakout Detected:** {breakout_type.replace('_', ' ').title()}")
                    
                    # Signal environment summary (without volume)
                    filter_count = sum([volatility == "normal", time_ok, 
                                      momentum_score >= 6 or momentum_score <= 4,
                                      price_profile["strength"] >= 1.2])
                    
                    if filter_count >= 3:
                        st.success(f"🌟 **Excellent Signal Environment** - {filter_count}/4 conditions favorable")
                    elif filter_count >= 2:
                        st.info(f"⚡ **Good Signal Environment** - {filter_count}/4 conditions favorable")
                    elif filter_count >= 1:
                        st.warning(f"⚠️ **Fair Signal Environment** - {filter_count}/4 conditions favorable")
                    else:
                        st.error(f"❌ **Poor Signal Environment** - {filter_count}/4 conditions favorable")
        else:
            st.error("No chart data available")
    
    with col2:
        st.header("Options Analysis")
        
        expiry_data = get_expiry_list()
        if expiry_data and 'data' in expiry_data:
            expiry_dates = expiry_data['data']
            selected_expiry = st.selectbox("Expiry", expiry_dates)
            
            underlying_price, option_summary = analyze_options(selected_expiry)
            
            if underlying_price and option_summary is not None:
                st.info(f"Spot: ₹{underlying_price:.2f}")
                st.dataframe(option_summary, use_container_width=True)
                
                if enable_signals and not df.empty and is_market_hours():
                    check_signals(df, option_summary, underlying_price, proximity)
                        
            else:
                st.error("Options data unavailable")
        else:
            st.error("Expiry data unavailable")
    
    # Footer
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"🕒 Last Updated: {current_time}")
    
    # Additional debugging info
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.sidebar.subheader("🔍 Debug Information")
        st.sidebar.text(f"Market Hours: {is_market_hours()}")
        st.sidebar.text(f"Current Time: {current_time}")
        st.sidebar.text(f"Bot Token Set: {'Yes' if TELEGRAM_BOT_TOKEN else 'No'}")
        st.sidebar.text(f"Chat ID Set: {'Yes' if TELEGRAM_CHAT_ID else 'No'}")
        
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            # Test bot connectivity
            if st.sidebar.button("🔍 Test Bot Connection"):
                test_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
                try:
                    response = requests.get(test_url, timeout=10)
                    if response.status_code == 200:
                        bot_info = response.json()
                        if bot_info.get('ok'):
                            bot_data = bot_info.get('result', {})
                            st.sidebar.success(f"✅ Bot '{bot_data.get('first_name', 'Unknown')}' is active")
                        else:
                            st.sidebar.error("❌ Bot token invalid")
                    else:
                        st.sidebar.error(f"❌ Bot API error: {response.status_code}")
                except Exception as e:
                    st.sidebar.error(f"❌ Connection error: {str(e)}")

if __name__ == "__main__":
    # Initialize session state
    if 'signal_log' not in st.session_state:
        st.session_state.signal_log = []
    if 'news_cache' not in st.session_state:
        st.session_state.news_cache = {}
    if 'alpha_vantage_news' not in st.session_state:
        st.session_state.alpha_vantage_news = []
    
    main()