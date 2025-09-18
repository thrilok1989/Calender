import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import numpy as np
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(page_title="Nifty Analyzer Enhanced", page_icon="📈", layout="wide")

# Constants
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"
VIX_SCRIP = 25  # India VIX security ID (need to verify from instrument list)

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "")

def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return market_start <= now <= market_end

if is_market_hours():
    st_autorefresh(interval=35000, key="refresh")

class DhanAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
    
    def get_intraday_data(self, security_id=NIFTY_SCRIP, interval="5", days_back=1, include_volume=True):
        url = "https://api.dhan.co/v2/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": NIFTY_SEG if security_id == NIFTY_SCRIP else "IDX_I",
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
    
    def get_market_quote(self, instruments):
        url = "https://api.dhan.co/v2/marketfeed/quote"
        try:
            response = requests.post(url, headers=self.headers, json=instruments)
            return response.json() if response.status_code == 200 else None
        except:
            return None

def get_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_expiry_list():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    clean_message = message.replace("<", "less than ").replace(">", "greater than ")
    
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": clean_message, "disable_web_page_preview": True}
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            st.success("✅ Telegram message sent!")
            return True
        else:
            st.error(f"❌ Telegram error: {response.json().get('description', 'Unknown')}")
    except Exception as e:
        st.error(f"❌ Network error: {str(e)}")
    return False

def process_candle_data(data):
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'timestamp': data['timestamp'],
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data.get('volume', [0] * len(data['open']))
    })
    
    ist = pytz.timezone('Asia/Kolkata')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
    return df

# Alpha Vantage Functions
def fetch_usd_inr_data():
    if not ALPHA_VANTAGE_KEY:
        return None
    
    try:
        current_time = time.time()
        if 'last_usd_inr_request' in st.session_state:
            if current_time - st.session_state.last_usd_inr_request < 1800:  # 30 minutes
                return st.session_state.get('usd_inr_data')
        
        url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=USD&to_symbol=INR&interval=5min&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if 'Time Series FX (5min)' in data:
                fx_data = data['Time Series FX (5min)']
                processed_data = []
                
                for timestamp, values in list(fx_data.items())[:50]:  # Last 50 data points
                    processed_data.append({
                        'datetime': pd.to_datetime(timestamp),
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close'])
                    })
                
                df = pd.DataFrame(processed_data)
                df = df.sort_values('datetime').reset_index(drop=True)
                
                st.session_state.last_usd_inr_request = current_time
                st.session_state.usd_inr_data = df
                return df
        
    except Exception as e:
        st.error(f"USD/INR fetch error: {e}")
    
    return st.session_state.get('usd_inr_data')

def fetch_alpha_vantage_news():
    if not ALPHA_VANTAGE_KEY:
        return []
    
    try:
        current_time = time.time()
        if 'last_news_request' in st.session_state:
            if current_time - st.session_state.last_news_request < 900:  # 15 minutes
                return st.session_state.get('alpha_vantage_news', [])
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=NIFTY&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            feed = data.get('feed', [])[:5]
            
            processed_news = []
            for item in feed:
                sentiment_score = item.get('overall_sentiment', {}).get('score', 0) if isinstance(item.get('overall_sentiment'), dict) else 0
                processed_news.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', '')[:200],
                    'sentiment_score': sentiment_score
                })
            
            st.session_state.last_news_request = current_time
            st.session_state.alpha_vantage_news = processed_news
            return processed_news
            
    except:
        pass
    
    return st.session_state.get('alpha_vantage_news', [])

def analyze_news_sentiment(news_items):
    if not news_items:
        return {"overall": "neutral", "score": 0, "bullish_count": 0, "bearish_count": 0}
    
    scores = []
    for item in news_items:
        score = item.get('sentiment_score', 0)
        if 0 <= score <= 1:
            scores.append(score * 2 - 1)
        else:
            scores.append(score)
    
    if scores:
        avg_score = sum(scores) / len(scores)
        overall = "bullish" if avg_score > 0.2 else "bearish" if avg_score < -0.2 else "neutral"
        
        return {
            "overall": overall,
            "score": round(avg_score, 3),
            "bullish_count": sum(1 for s in scores if s > 0.2),
            "bearish_count": sum(1 for s in scores if s < -0.2),
            "confidence": min(abs(avg_score) * 2, 1.0)
        }
    
    return {"overall": "neutral", "score": 0, "bullish_count": 0, "bearish_count": 0}

# Technical Analysis Functions
def calculate_rsi(df, periods=14):
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

def calculate_fibonacci_levels(df, lookback=50):
    if len(df) < lookback:
        return {}
    
    recent_data = df.tail(lookback)
    high = recent_data['high'].max()
    low = recent_data['low'].min()
    diff = high - low
    
    return {
        'high': high,
        'low': low,
        '23.6%': high - (diff * 0.236),
        '38.2%': high - (diff * 0.382),
        '50%': high - (diff * 0.5),
        '61.8%': high - (diff * 0.618),
        '78.6%': high - (diff * 0.786)
    }

def find_pivots(prices, length, pivot_type='high'):
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
        else:
            if current < left_side.min() and current < right_side.min():
                pivots.iloc[i] = current
                
    return pivots

def get_market_trend(df):
    if len(df) < 20:
        return "neutral"
    
    sma_short = df['close'].rolling(10).mean().iloc[-1]
    sma_long = df['close'].rolling(20).mean().iloc[-1]
    
    if sma_short > sma_long * 1.002:
        return "bullish"
    elif sma_short < sma_long * 0.998:
        return "bearish"
    else:
        return "neutral"

def get_momentum_score(df):
    if len(df) < 15:
        return 5
    
    rsi_values = calculate_rsi(df)
    latest_rsi = rsi_values.iloc[-1]
    return max(1, min(10, int(latest_rsi / 10))) if not pd.isna(latest_rsi) else 5

def check_pcr_condition(pcr, signal_type):
    if signal_type == "bullish":
        return pcr > 1.3
    elif signal_type == "bearish":
        return pcr < 0.7
    return False

def get_pcr_condition_text(pcr, signal_type):
    if signal_type == "bullish":
        condition = "above" if pcr > 1.3 else "below"
        return f"PCR {pcr:.2f} ({condition} 1.3 for bull)"
    elif signal_type == "bearish":
        condition = "below" if pcr < 0.7 else "above"
        return f"PCR {pcr:.2f} ({condition} 0.7 for bear)"
    return f"PCR {pcr:.2f}"

# Options Analysis Functions
def calculate_max_pain(option_data):
    if option_data is None or 'data' not in option_data:
        return None
    
    oc_data = option_data['data']['oc']
    max_pain_data = {}
    
    for strike, data in oc_data.items():
        strike_price = float(strike)
        ce_oi = data.get('ce', {}).get('oi', 0) if 'ce' in data else 0
        pe_oi = data.get('pe', {}).get('oi', 0) if 'pe' in data else 0
        total_oi = ce_oi + pe_oi
        max_pain_data[strike_price] = total_oi
    
    if max_pain_data:
        return max(max_pain_data, key=max_pain_data.get)
    return None

def get_gamma_levels(option_data):
    if option_data is None or 'data' not in option_data:
        return []
    
    oc_data = option_data['data']['oc']
    gamma_levels = []
    
    for strike, data in oc_data.items():
        strike_price = float(strike)
        ce_gamma = data.get('ce', {}).get('greeks', {}).get('gamma', 0) if 'ce' in data else 0
        pe_gamma = data.get('pe', {}).get('greeks', {}).get('gamma', 0) if 'pe' in data else 0
        total_gamma = abs(ce_gamma) + abs(pe_gamma)
        
        if total_gamma > 0.001:  # Significant gamma threshold
            gamma_levels.append({'strike': strike_price, 'gamma': total_gamma})
    
    return sorted(gamma_levels, key=lambda x: x['gamma'], reverse=True)[:5]

def analyze_iv_rank(option_data):
    if option_data is None or 'data' not in option_data:
        return {"current_iv": 0, "rank": "unknown"}
    
    oc_data = option_data['data']['oc']
    iv_values = []
    
    for strike, data in oc_data.items():
        if 'ce' in data and data['ce'].get('implied_volatility'):
            iv_values.append(data['ce']['implied_volatility'])
        if 'pe' in data and data['pe'].get('implied_volatility'):
            iv_values.append(data['pe']['implied_volatility'])
    
    if iv_values:
        avg_iv = sum(iv_values) / len(iv_values)
        # Simple classification without historical data
        if avg_iv > 20:
            rank = "high"
        elif avg_iv < 10:
            rank = "low"
        else:
            rank = "medium"
        
        return {"current_iv": avg_iv, "rank": rank}
    
    return {"current_iv": 0, "rank": "unknown"}

def create_enhanced_chart(df, usd_inr_df=None, title="Nifty Analysis"):
    if df.empty:
        return go.Figure()
    
    # Filter to market hours
    df_filtered = df[
        (df['datetime'].dt.weekday < 5) &
        (df['datetime'].dt.time >= pd.Timestamp('09:15').time()) &
        (df['datetime'].dt.time <= pd.Timestamp('15:30').time())
    ]
    
    if df_filtered.empty:
        return go.Figure()
    
    # Create subplots - fix the None check
    has_usd_inr = usd_inr_df is not None and len(usd_inr_df) > 0 if isinstance(usd_inr_df, pd.DataFrame) else False
    rows = 4 if has_usd_inr else 3
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.5, 0.15, 0.15, 0.2] if rows == 4 else [0.6, 0.2, 0.2],
        subplot_titles=("Price + Fibonacci", "Volume", "RSI (14)", "USD/INR") if rows == 4 else ("Price + Fibonacci", "Volume", "RSI (14)")
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df_filtered['datetime'], 
        open=df_filtered['open'], 
        high=df_filtered['high'], 
        low=df_filtered['low'], 
        close=df_filtered['close'], 
        name='Nifty',
        increasing_line_color='#00ff88', 
        decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Add Fibonacci levels
    fib_levels = calculate_fibonacci_levels(df_filtered)
    if fib_levels:
        x_start, x_end = df_filtered['datetime'].min(), df_filtered['datetime'].max()
        
        fib_colors = {'23.6%': '#FF6B6B', '38.2%': '#4ECDC4', '50%': '#45B7D1', '61.8%': '#96CEB4', '78.6%': '#FFEAA7'}
        
        for level, price in fib_levels.items():
            if level in fib_colors:
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end, y0=price, y1=price,
                    line=dict(color=fib_colors[level], width=1, dash="dot"),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=x_end, y=price, text=f"Fib {level}: {price:.1f}",
                    showarrow=False, xshift=10, font=dict(size=8, color=fib_colors[level]),
                    row=1, col=1
                )
    
    # Volume
    if 'volume' in df_filtered.columns and df_filtered['volume'].sum() > 0:
        volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                        for close, open in zip(df_filtered['close'], df_filtered['open'])]
        fig.add_trace(go.Bar(
            x=df_filtered['datetime'], y=df_filtered['volume'], name='Volume',
            marker_color=volume_colors, opacity=0.7
        ), row=2, col=1)
    
    # RSI
    if len(df_filtered) > 14:
        rsi_values = calculate_rsi(df_filtered)
        fig.add_trace(go.Scatter(
            x=df_filtered['datetime'], y=rsi_values, name='RSI',
            line=dict(color='#ffaa00', width=2), mode='lines'
        ), row=3, col=1)
        
        for level, color, label in [(70, "red", "Overbought"), (30, "green", "Oversold"), (50, "gray", "Midline")]:
            fig.add_hline(y=level, line_dash="dash" if level != 50 else "dot", 
                         line_color=color, annotation_text=f"{label} ({level})", row=3, col=1)
        
        fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    # USD/INR chart
    if rows == 4 and has_usd_inr:
        fig.add_trace(go.Candlestick(
            x=usd_inr_df['datetime'], 
            open=usd_inr_df['open'], 
            high=usd_inr_df['high'], 
            low=usd_inr_df['low'], 
            close=usd_inr_df['close'], 
            name='USD/INR',
            increasing_line_color='#FFA500', 
            decreasing_line_color='#8A2BE2'
        ), row=4, col=1)
    
    fig.update_layout(
        title=title, 
        template='plotly_dark', 
        height=800 if rows == 4 else 700,
        xaxis_rangeslider_visible=False, 
        showlegend=False
    )
    
    # Update x-axis labels
    for i in range(1, rows):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    fig.update_xaxes(showticklabels=True, row=rows, col=1)
    
    return fig

def analyze_options(expiry):
    option_data = get_option_chain(expiry)
    if not option_data or 'data' not in option_data:
        return None, None, None, None
    
    data = option_data['data']
    underlying = data['last_price']
    oc_data = data['oc']
    
    # Process options data
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
    
    # Rename columns for compatibility
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
    
    # Calculate additional analytics
    max_pain = calculate_max_pain(option_data)
    gamma_levels = get_gamma_levels(option_data)
    iv_analysis = analyze_iv_rank(option_data)
    
    return underlying, pd.DataFrame(bias_results), max_pain, gamma_levels, iv_analysis

def check_enhanced_signals(df, option_data, current_price, max_pain, gamma_levels, iv_analysis, usd_inr_df=None):
    if df.empty or option_data is None or not current_price:
        return
    
    # Get analysis data
    news_items = fetch_alpha_vantage_news()
    news_sentiment = analyze_news_sentiment(news_items)
    market_trend = get_market_trend(df)
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return
    
    row = atm_data.iloc[0]
    rsi_values = calculate_rsi(df)
    current_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 and not pd.isna(rsi_values.iloc[-1]) else 50
    
    # Check USD/INR correlation
    usd_inr_direction = ""
    if usd_inr_df is not None and not usd_inr_df.empty and len(usd_inr_df) > 1:
        usd_inr_change = usd_inr_df['close'].iloc[-1] - usd_inr_df['close'].iloc[-2]
        usd_inr_direction = f"USD/INR {'Rising' if usd_inr_change > 0 else 'Falling'} ({usd_inr_change:+.4f})"
    
    # Fibonacci levels
    fib_levels = calculate_fibonacci_levels(df)
    fib_support = None
    fib_resistance = None
    
    if fib_levels:
        for level, price in fib_levels.items():
            if level in ['38.2%', '50%', '61.8%']:
                if abs(current_price - price) < 10:  # Within 10 points
                    if price < current_price:
                        fib_support = f"Fib {level}: {price:.1f}"
                    else:
                        fib_resistance = f"Fib {level}: {price:.1f}"
    
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
    
    # Enhanced signal conditions
    max_pain_info = f"Max Pain: {max_pain:.0f}" if max_pain else "Max Pain: N/A"
    gamma_info = f"Top Gamma: {gamma_levels[0]['strike']:.0f}" if gamma_levels else "Gamma: N/A"
    iv_info = f"IV Rank: {iv_analysis['rank'].title()} ({iv_analysis['current_iv']:.1f}%)"
    
    # Comprehensive bias info
    comprehensive_info = f"""
📊 ENHANCED ANALYSIS 📊

🔹 OPTIONS ANALYTICS:
• {max_pain_info}
• {gamma_info}
• {iv_info}
• PCR: {pcr:.2f}
• Level: {row['Level']}

🔹 FIBONACCI LEVELS:
• Support: {fib_support if fib_support else 'None nearby'}
• Resistance: {fib_resistance if fib_resistance else 'None nearby'}

🔹 TECHNICAL STATUS:
• RSI: {current_rsi:.1f}
• Trend: {market_trend.title()}
• {news_emoji} News: {news_sentiment['overall'].title()}

🔹 CORRELATIONS:
• {usd_inr_direction if usd_inr_direction else 'USD/INR: N/A'}

🔹 QUANTITATIVE DATA:
• CE OI: {row.get('openInterest_CE', 0):,}
• PE OI: {row.get('openInterest_PE', 0):,}
• CE Volume: {row.get('totalTradedVolume_CE', 0):,}
• PE Volume: {row.get('totalTradedVolume_PE', 0):,}
• Strike: {row['Strike']}
"""
    
    # Enhanced Primary Signal
    if bias_aligned_bullish and pcr_bullish:
        signal_type = "CALL"
        pcr_text = get_pcr_condition_text(pcr, "bullish")
        
        # Check confluence with other factors
        confluence_factors = []
        if max_pain and current_price > max_pain:
            confluence_factors.append("Above Max Pain")
        if fib_support:
            confluence_factors.append(f"Near {fib_support}")
        if iv_analysis['rank'] == 'low':
            confluence_factors.append("Low IV Environment")
        
        confluence_text = f"\nConfluence: {', '.join(confluence_factors)}" if confluence_factors else ""
        
        message = f"""
🚨 ENHANCED PRIMARY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.1f}
📈 {pcr_text}

{news_emoji} Market Sentiment: {news_sentiment['overall'].title()}
{max_pain_info} | {gamma_info}
{iv_info}{confluence_text}

All Bias Aligned + Enhanced Analytics Confirm

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_info}
"""
        send_telegram(message)
        st.success(f"ENHANCED PRIMARY {signal_type} signal sent!")
        
    elif bias_aligned_bearish and pcr_bearish:
        signal_type = "PUT"
        pcr_text = get_pcr_condition_text(pcr, "bearish")
        
        # Check confluence with other factors
        confluence_factors = []
        if max_pain and current_price < max_pain:
            confluence_factors.append("Below Max Pain")
        if fib_resistance:
            confluence_factors.append(f"Near {fib_resistance}")
        if iv_analysis['rank'] == 'low':
            confluence_factors.append("Low IV Environment")
        
        confluence_text = f"\nConfluence: {', '.join(confluence_factors)}" if confluence_factors else ""
        
        message = f"""
🚨 ENHANCED PRIMARY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 RSI: {current_rsi:.1f}
📈 {pcr_text}

{news_emoji} Market Sentiment: {news_sentiment['overall'].title()}
{max_pain_info} | {gamma_info}
{iv_info}{confluence_text}

All Bias Aligned + Enhanced Analytics Confirm

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}

{comprehensive_info}
"""
        send_telegram(message)
        st.success(f"ENHANCED PRIMARY {signal_type} signal sent!")

def main():
    st.title("📈 Enhanced Nifty Analyzer")
    
    # Market status
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"⚠️ Market closed. Time: {current_time.strftime('%H:%M:%S IST')}")
    
    # Sidebar
    st.sidebar.header("🎛️ Enhanced Settings")
    interval = st.sidebar.selectbox("Timeframe", ["1", "3", "5", "10", "15"], index=2)
    enable_signals = st.sidebar.checkbox("Enable Enhanced Signals", value=True)
    enable_usd_inr = st.sidebar.checkbox("Show USD/INR Chart", value=True)
    
    # Telegram Setup
    st.sidebar.subheader("📱 Telegram")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        st.sidebar.success("✅ Configured")
        if st.sidebar.button("📤 Test Message"):
            test_msg = f"Enhanced Nifty Analyzer Test\nTime: {current_time.strftime('%H:%M:%S IST')}\nAll systems operational!"
            send_telegram(test_msg)
    else:
        st.sidebar.error("❌ Not configured")
    
    # Main content
    api = DhanAPI()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Enhanced Chart Analysis")
        
        # Get data
        data = api.get_intraday_data(NIFTY_SCRIP, interval)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        # Get VIX data
        vix_data = api.get_intraday_data(VIX_SCRIP, interval) if VIX_SCRIP else None
        vix_df = process_candle_data(vix_data) if vix_data else pd.DataFrame()
        
        # Get USD/INR data
        usd_inr_df = fetch_usd_inr_data() if enable_usd_inr else None
        
        # Get current price
        current_price = None
        vix_price = None
        
        if not df.empty:
            current_price = df['close'].iloc[-1]
        if not vix_df.empty:
            vix_price = vix_df['close'].iloc[-1]
        
        # Display metrics
        if not df.empty and len(df) > 1:
            prev_close = df['close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            col1_m, col2_m, col3_m, col4_m = st.columns(4)
            with col1_m:
                st.metric("Nifty", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            with col2_m:
                st.metric("High", f"₹{df['high'].max():,.2f}")
            with col3_m:
                st.metric("Low", f"₹{df['low'].min():,.2f}")
            with col4_m:
                if vix_price:
                    st.metric("VIX", f"{vix_price:.2f}")
                else:
                    st.metric("VIX", "N/A")
        
        # Enhanced chart
        if not df.empty:
            fig = create_enhanced_chart(df, usd_inr_df, f"Enhanced Nifty {interval}min")
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced market analysis
            if len(df) > 20:
                st.subheader("📊 Enhanced Market Analysis")
                
                # Get all analysis data
                fib_levels = calculate_fibonacci_levels(df)
                market_trend = get_market_trend(df)
                momentum_score = get_momentum_score(df)
                rsi_values = calculate_rsi(df)
                current_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 and not pd.isna(rsi_values.iloc[-1]) else 50
                
                # Display analysis
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    trend_color = "🟢" if market_trend == "bullish" else "🔴" if market_trend == "bearish" else "🟡"
                    st.metric("Trend", f"{trend_color} {market_trend.title()}")
                with col_b:
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI", f"{rsi_color} {current_rsi:.1f}")
                with col_c:
                    momentum_color = "🟢" if momentum_score >= 6 else "🔴" if momentum_score <= 4 else "🟡"
                    st.metric("Momentum", f"{momentum_color} {momentum_score}/10")
                with col_d:
                    if vix_price:
                        vix_color = "🔴" if vix_price > 20 else "🟢" if vix_price < 15 else "🟡"
                        st.metric("Fear Index", f"{vix_color} {vix_price:.1f}")
                    else:
                        st.metric("Fear Index", "N/A")
                
                # Fibonacci levels display
                if fib_levels:
                    st.subheader("🔄 Fibonacci Retracement Levels")
                    fib_col1, fib_col2 = st.columns(2)
                    with fib_col1:
                        st.write(f"**Swing High:** ₹{fib_levels['high']:.2f}")
                        st.write(f"**23.6%:** ₹{fib_levels['23.6%']:.2f}")
                        st.write(f"**38.2%:** ₹{fib_levels['38.2%']:.2f}")
                    with fib_col2:
                        st.write(f"**Swing Low:** ₹{fib_levels['low']:.2f}")
                        st.write(f"**50%:** ₹{fib_levels['50%']:.2f}")
                        st.write(f"**61.8%:** ₹{fib_levels['61.8%']:.2f}")
                
                # News analysis
                news_items = fetch_alpha_vantage_news()
                news_sentiment = analyze_news_sentiment(news_items)
                
                if news_items:
                    st.subheader("📰 Market Sentiment")
                    sentiment_color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(news_sentiment["overall"], "🟡")
                    
                    col_news1, col_news2, col_news3 = st.columns(3)
                    with col_news1:
                        st.metric("News Sentiment", f"{sentiment_color} {news_sentiment['overall'].title()}")
                    with col_news2:
                        st.metric("Bullish Items", news_sentiment["bullish_count"])
                    with col_news3:
                        st.metric("Bearish Items", news_sentiment["bearish_count"])
        else:
            st.error("No chart data available")
    
    with col2:
        st.header("Enhanced Options")
        
        # Get expiry list
        expiry_data = get_expiry_list()
        if expiry_data and 'data' in expiry_data:
            selected_expiry = st.selectbox("Expiry", expiry_data['data'])
            
            # Analyze options
            result = analyze_options(selected_expiry)
            if result and len(result) == 5:
                underlying_price, option_summary, max_pain, gamma_levels, iv_analysis = result
                
                if underlying_price and option_summary is not None:
                    st.info(f"Spot: ₹{underlying_price:.2f}")
                    
                    # Enhanced options metrics
                    col_opt1, col_opt2 = st.columns(2)
                    with col_opt1:
                        if max_pain:
                            pain_diff = underlying_price - max_pain
                            pain_color = "🟢" if abs(pain_diff) < 50 else "🟡"
                            st.metric("Max Pain", f"{pain_color} {max_pain:.0f}", f"{pain_diff:+.0f}")
                        else:
                            st.metric("Max Pain", "N/A")
                    
                    with col_opt2:
                        iv_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(iv_analysis['rank'], "🟡")
                        st.metric("IV Rank", f"{iv_color} {iv_analysis['rank'].title()}", f"{iv_analysis['current_iv']:.1f}%")
                    
                    # Gamma levels
                    if gamma_levels:
                        st.subheader("⚡ Top Gamma Levels")
                        for i, level in enumerate(gamma_levels[:3]):
                            st.write(f"{i+1}. Strike {level['strike']:.0f}: Gamma {level['gamma']:.4f}")
                    
                    # Options summary table
                    st.dataframe(option_summary, use_container_width=True)
                    
                    # Enhanced signals
                    if enable_signals and not df.empty and is_market_hours() and current_price:
                        check_enhanced_signals(df, option_summary, current_price, max_pain, gamma_levels, iv_analysis, usd_inr_df)
                else:
                    st.error("Options analysis failed")
            else:
                st.error("Options data unavailable")
        else:
            st.error("Expiry list unavailable")
    
    # Footer
    st.sidebar.info(f"🕒 Updated: {current_time.strftime('%H:%M:%S IST')}")

if __name__ == "__main__":
    # Initialize session state
    for key in ['signal_log', 'news_cache', 'alpha_vantage_news', 'usd_inr_data']:
        if key not in st.session_state:
            st.session_state[key] = [] if 'news' in key else {}
    
    main()