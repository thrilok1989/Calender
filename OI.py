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
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

class DhanAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
    
    def get_intraday_data(self, interval="5", days_back=1):
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
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = {NIFTY_SEG: [NIFTY_SCRIP]}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
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
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def process_candle_data(data):
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

def find_pivot_highs_proper(highs, length):
    """
    Proper pivot high detection - checks if bar is highest among 'length' bars on BOTH sides
    """
    if len(highs) < length * 2 + 1:
        return pd.Series(index=highs.index, dtype=float)
    
    pivot_highs = pd.Series(index=highs.index, dtype=float)
    
    for i in range(length, len(highs) - length):
        current_high = highs.iloc[i]
        
        # Check 'length' bars to the left and right
        left_side = highs.iloc[i-length:i]
        right_side = highs.iloc[i+1:i+length+1]
        
        # Current bar must be strictly higher than all bars on both sides
        if (current_high > left_side.max()) and (current_high > right_side.max()):
            pivot_highs.iloc[i] = current_high
            
    return pivot_highs

def find_pivot_lows_proper(lows, length):
    """
    Proper pivot low detection - checks if bar is lowest among 'length' bars on BOTH sides
    """
    if len(lows) < length * 2 + 1:
        return pd.Series(index=lows.index, dtype=float)
        
    pivot_lows = pd.Series(index=lows.index, dtype=float)
    
    for i in range(length, len(lows) - length):
        current_low = lows.iloc[i]
        
        # Check 'length' bars to the left and right
        left_side = lows.iloc[i-length:i]
        right_side = lows.iloc[i+1:i+length+1]
        
        # Current bar must be strictly lower than all bars on both sides
        if (current_low < left_side.min()) and (current_low < right_side.min()):
            pivot_lows.iloc[i] = current_low
            
    return pivot_lows

def detect_level_touches(df, pivot_value, tolerance_pct=0.09):
    """
    Detect when price touches a pivot level with tolerance
    """
    if df.empty:
        return []
        
    touches = []
    tolerance = pivot_value * (tolerance_pct / 100)
    
    for i, row in df.iterrows():
        # Check if high touched the level
        if abs(row['high'] - pivot_value) <= tolerance:
            touches.append({
                'datetime': row['datetime'],
                'price': pivot_value,
                'touch_type': 'high_touch',
                'actual_price': row['high'],
                'bar_index': i
            })
        
        # Check if low touched the level  
        elif abs(row['low'] - pivot_value) <= tolerance:
            touches.append({
                'datetime': row['datetime'],
                'price': pivot_value,
                'touch_type': 'low_touch', 
                'actual_price': row['low'],
                'bar_index': i
            })
            
    return touches

# ===== ENHANCED SIGNAL FILTERS =====

def get_market_trend(df, period=20):
    """Market trend filter"""
    if len(df) < period:
        return "neutral"
    
    sma_short = df['close'].rolling(10).mean().iloc[-1]
    sma_long = df['close'].rolling(20).mean().iloc[-1]
    
    if sma_short > sma_long * 1.002:  # 0.2% threshold
        return "bullish"
    elif sma_short < sma_long * 0.998:
        return "bearish"
    else:
        return "neutral"

def check_volume_confirmation(df, lookback=10):
    """Volume confirmation filter"""
    if len(df) < lookback:
        return False
    
    recent_volume = df['volume'].iloc[-3:].mean()
    avg_volume = df['volume'].rolling(lookback).mean().iloc[-1]
    
    return recent_volume > avg_volume * 1.2  # 20% above average

def check_pivot_confluence(df, current_price, proximity=5):
    """Multiple timeframe confluence filter"""
    timeframes = ["5", "10", "15"]
    support_count = 0
    resistance_count = 0
    
    for tf in timeframes:
        nearby = get_nearby_pivot_levels(df, current_price, proximity)
        for level in nearby:
            if level['timeframe'] == tf:
                if level['type'] == 'support':
                    support_count += 1
                elif level['type'] == 'resistance':
                    resistance_count += 1
    
    return support_count >= 2 or resistance_count >= 2

def check_options_strength(option_data):
    """Strong options flow filter"""
    if option_data is None or option_data.empty:
        return False
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return False
    
    atm = atm_data.iloc[0]
    
    ce_chg = abs(atm.get('changeinOpenInterest_CE', 0))
    pe_chg = abs(atm.get('changeinOpenInterest_PE', 0))
    
    # Require stronger OI changes
    min_oi_change = 500  # Minimum 500 contracts change
    strong_dominance = max(ce_chg, pe_chg) > min_oi_change
    
    return strong_dominance

def get_market_volatility(df, period=14):
    """Market volatility filter"""
    if len(df) < period:
        return "normal"
    
    returns = df['close'].pct_change().dropna()
    volatility = returns.rolling(period).std().iloc[-1] * 100
    
    if volatility > 2.0:  # > 2% daily volatility
        return "high"
    elif volatility < 0.5:  # < 0.5% daily volatility  
        return "low"
    else:
        return "normal"

def is_good_signal_time():
    """Time-based filter"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Avoid first 15 minutes and last 30 minutes
    market_start = now.replace(hour=9, minute=15, second=0)
    market_end = now.replace(hour=15, minute=15, second=0)
    
    return market_start <= now <= market_end

def calculate_signal_strength(volume_ok, confluence_ok, options_strong, volatility_normal, time_ok, trend_aligned):
    """Calculate signal strength score 1-10"""
    filters_passed = sum([volume_ok, confluence_ok, options_strong, volatility_normal, time_ok, trend_aligned])
    
    # Base score from filters (max 6)
    base_score = filters_passed
    
    # Bonus points for strong confluence and options
    bonus = 0
    if confluence_ok and options_strong:
        bonus += 2
    if volume_ok and trend_aligned:
        bonus += 1
    
    return min(base_score + bonus, 10)

# ===== END ENHANCED FILTERS =====

def get_pivots(df, timeframe="5", length=4):
    """
    Enhanced pivot detection with proper algorithm and non-repainting protection
    """
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
        
        # Find all pivot highs and lows using proper algorithm
        pivot_highs = find_pivot_highs_proper(resampled['high'], length)
        pivot_lows = find_pivot_lows_proper(resampled['low'], length)
        
        pivots = []
        
        # Collect pivot highs (exclude last few to avoid repainting)
        valid_pivot_highs = pivot_highs.dropna()
        if len(valid_pivot_highs) > 1:
            # Keep all but the most recent to avoid repainting
            for timestamp, value in valid_pivot_highs[:-1].items():
                pivots.append({
                    'type': 'high', 
                    'timeframe': timeframe, 
                    'timestamp': timestamp, 
                    'value': float(value),
                    'confirmed': True
                })
            
            # Add the most recent but mark as unconfirmed
            if len(valid_pivot_highs) >= 1:
                last_high = valid_pivot_highs.iloc[-1]
                last_high_time = valid_pivot_highs.index[-1]
                pivots.append({
                    'type': 'high',
                    'timeframe': timeframe,
                    'timestamp': last_high_time,
                    'value': float(last_high),
                    'confirmed': False
                })
        
        # Collect pivot lows (exclude last few to avoid repainting)
        valid_pivot_lows = pivot_lows.dropna()
        if len(valid_pivot_lows) > 1:
            # Keep all but the most recent to avoid repainting
            for timestamp, value in valid_pivot_lows[:-1].items():
                pivots.append({
                    'type': 'low',
                    'timeframe': timeframe, 
                    'timestamp': timestamp,
                    'value': float(value),
                    'confirmed': True
                })
            
            # Add the most recent but mark as unconfirmed
            if len(valid_pivot_lows) >= 1:
                last_low = valid_pivot_lows.iloc[-1] 
                last_low_time = valid_pivot_lows.index[-1]
                pivots.append({
                    'type': 'low',
                    'timeframe': timeframe,
                    'timestamp': last_low_time, 
                    'value': float(last_low),
                    'confirmed': False
                })
        
        return pivots
        
    except Exception as e:
        print(f"Error in pivot calculation: {e}")
        return []

def get_nearby_pivot_levels(df, current_price, proximity=5.0):
    """
    Get confirmed pivot levels near current price for signal generation
    """
    if df.empty:
        return []
        
    nearby_levels = []
    timeframes = ["5", "10", "15"]
    
    for timeframe in timeframes:
        pivots = get_pivots(df, timeframe, length=4)
        
        for pivot in pivots:
            # Only use confirmed pivots for signals to avoid repainting
            if not pivot.get('confirmed', True):
                continue
                
            distance = abs(current_price - pivot['value'])
            if distance <= proximity:
                level_type = 'resistance' if pivot['type'] == 'high' else 'support'
                nearby_levels.append({
                    'type': level_type,
                    'pivot_type': pivot['type'], 
                    'value': pivot['value'],
                    'timeframe': timeframe,
                    'distance': distance,
                    'timestamp': pivot['timestamp'],
                    'confirmed': pivot['confirmed']
                })
    
    # Sort by distance (closest first)
    nearby_levels.sort(key=lambda x: x['distance'])
    return nearby_levels

def create_chart(df, title):
    """
    Enhanced chart with proper pivot levels, shadows, and labels
    """
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Add volume
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)
    
    # Add enhanced pivot levels
    if len(df) > 50:
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", '#4444ff']
        
        x_start, x_end = df['datetime'].min(), df['datetime'].max()
        
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            
            # Get recent pivots only (last 5 of each type)
            recent_highs = [p for p in pivots if p['type'] == 'high'][-5:]
            recent_lows = [p for p in pivots if p['type'] == 'low'][-5:]
            
            # Add pivot high lines
            for pivot in recent_highs:
                line_style = "solid" if pivot.get('confirmed', True) else "dash"
                line_width = 2 if pivot.get('confirmed', True) else 1
                
                # Main pivot line
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=line_width, dash=line_style),
                    row=1, col=1
                )
                
                # Shadow line (wider, more transparent)
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=5),
                    opacity=0.15 if pivot.get('confirmed', True) else 0.08,
                    row=1, col=1
                )
                
                # Add label
                status = "✓" if pivot.get('confirmed', True) else "?"
                fig.add_annotation(
                    x=x_end, y=pivot['value'],
                    text=f"{tf}M H {status}: {pivot['value']:.1f}",
                    showarrow=False, xshift=20,
                    font=dict(size=9, color=color),
                    row=1, col=1
                )
            
            # Add pivot low lines
            for pivot in recent_lows:
                line_style = "solid" if pivot.get('confirmed', True) else "dash"
                line_width = 2 if pivot.get('confirmed', True) else 1
                
                # Main pivot line
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=line_width, dash=line_style),
                    row=1, col=1
                )
                
                # Shadow line
                fig.add_shape(
                    type="line", x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=5),
                    opacity=0.15 if pivot.get('confirmed', True) else 0.08,
                    row=1, col=1
                )
                
                # Add label
                status = "✓" if pivot.get('confirmed', True) else "?"
                fig.add_annotation(
                    x=x_end, y=pivot['value'],
                    text=f"{tf}M L {status}: {pivot['value']:.1f}",
                    showarrow=False, xshift=20,
                    font=dict(size=9, color=color),
                    row=1, col=1
                )
    
    fig.update_layout(title=title, template='plotly_dark', height=600,
                     xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def analyze_options(expiry):
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
    
    bias_results = []
    for _, row in df_filtered.iterrows():
        chg_oi_bias = "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish"
        volume_bias = "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish"
        
        ask_ce = row.get('askQty_CE', 0)
        ask_pe = row.get('askQty_PE', 0)
        bid_ce = row.get('bidQty_CE', 0)
        bid_pe = row.get('bidQty_PE', 0)
        
        ask_bias = "Bearish" if ask_ce > ask_pe else "Bullish"
        bid_bias = "Bullish" if bid_ce > bid_pe else "Bearish"
        
        ce_oi = row['openInterest_CE']
        pe_oi = row['openInterest_PE']
        level = "Support" if pe_oi > 1.12 * ce_oi else "Resistance" if ce_oi > 1.12 * pe_oi else "Neutral"
        
        bias_results.append({
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": level,
            "ChgOI_Bias": chg_oi_bias,
            "Volume_Bias": volume_bias,
            "Ask_Bias": ask_bias,
            "Bid_Bias": bid_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else 0, 2),
            "changeinOpenInterest_CE": row['changeinOpenInterest_CE'],
            "changeinOpenInterest_PE": row['changeinOpenInterest_PE']
        })
    
    return underlying, pd.DataFrame(bias_results)

def check_signals(df, option_data, current_price, proximity=5):
    if df.empty or option_data is None or not current_price:
        return
    
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
    
    # ===== EXISTING SIGNALS (UNCHANGED) =====
    
    # PRIMARY SIGNAL - Using enhanced pivot detection
    nearby_levels = get_nearby_pivot_levels(df, current_price, proximity)
    near_pivot = len(nearby_levels) > 0
    pivot_level = nearby_levels[0] if nearby_levels else None
    
    if near_pivot and pivot_level:
        primary_bullish_signal = (row['Level'] == 'Support' and bias_aligned_bullish and pivot_level['type'] == 'support')
        primary_bearish_signal = (row['Level'] == 'Resistance' and bias_aligned_bearish and pivot_level['type'] == 'resistance')
        
        if primary_bullish_signal or primary_bearish_signal:
            signal_type = "CALL" if primary_bullish_signal else "PUT"
            price_diff = current_price - pivot_level['value']
            
            # Check for recent touches to add confidence
            touches = detect_level_touches(df, pivot_level['value'])
            touch_info = f" (Touches: {len(touches)})" if touches else ""
            
            message = f"""
🚨 PRIMARY NIFTY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M {pivot_level['type'].title()} at ₹{pivot_level['value']:.2f}{touch_info}
🎯 ATM: {row['Strike']}

Conditions: {row['Level']}, All Bias Aligned, Confirmed Pivot
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success(f"🔔 PRIMARY {signal_type} signal sent!")
    
    # SECONDARY SIGNAL
    put_dominance = pe_chg_oi > 1.3 * ce_chg_oi if ce_chg_oi > 0 else False
    call_dominance = ce_chg_oi > 1.3 * pe_chg_oi if pe_chg_oi > 0 else False
    
    secondary_bullish_signal = (bias_aligned_bullish and put_dominance)
    secondary_bearish_signal = (bias_aligned_bearish and call_dominance)
    
    if secondary_bullish_signal or secondary_bearish_signal:
        signal_type = "CALL" if secondary_bullish_signal else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if secondary_bullish_signal and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        message = f"""
⚡ SECONDARY NIFTY {signal_type} SIGNAL - OI DOMINANCE ⚡

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

🔥 OI Dominance: {'PUT' if secondary_bullish_signal else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
📊 All Bias Aligned: {row['ChgOI_Bias']}, {row['Volume_Bias']}, {row['Ask_Bias']}, {row['Bid_Bias']}

ChgOI: CE {ce_chg_oi:,} | PE {pe_chg_oi:,}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success(f"⚡ SECONDARY {signal_type} signal sent!")
    
    # FOURTH SIGNAL - ALL BIAS ALIGNED
    if bias_aligned_bullish or bias_aligned_bearish:
        signal_type = "CALL" if bias_aligned_bullish else "PUT"
        
        message = f"""
🎯 FOURTH SIGNAL - ALL BIAS ALIGNED {signal_type} 🎯

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

All ATM Biases Aligned: {signal_type}
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}

ChgOI: CE {ce_chg_oi:,} | PE {pe_chg_oi:,}
PCR: {row['PCR']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success(f"🎯 FOURTH {signal_type} signal sent!")
    
    # ===== NEW ENHANCED SIGNALS =====
    
    # Calculate all filters
    market_trend = get_market_trend(df)
    volume_ok = check_volume_confirmation(df)
    confluence_ok = check_pivot_confluence(df, current_price, proximity)
    options_strong = check_options_strength(option_data)
    volatility = get_market_volatility(df)
    volatility_normal = volatility == "normal"
    time_ok = is_good_signal_time()
    
    # Trend alignment for different signal directions
    trend_aligned_bullish = market_trend in ["bullish", "neutral"]
    trend_aligned_bearish = market_trend in ["bearish", "neutral"]
    
    # FIFTH SIGNAL - ENHANCED PRIMARY WITH ALL FILTERS
    if near_pivot and pivot_level:
        enhanced_bullish = (
            primary_bullish_signal and
            trend_aligned_bullish and
            volume_ok and
            confluence_ok and
            options_strong and
            volatility_normal and
            time_ok
        )
        
        enhanced_bearish = (
            primary_bearish_signal and
            trend_aligned_bearish and
            volume_ok and
            confluence_ok and
            options_strong and
            volatility_normal and
            time_ok
        )
        
        if enhanced_bullish or enhanced_bearish:
            signal_type = "CALL" if enhanced_bullish else "PUT"
            price_diff = current_price - pivot_level['value']
            
            # Calculate signal strength
            signal_strength = calculate_signal_strength(
                volume_ok, confluence_ok, options_strong, 
                volatility_normal, time_ok, 
                trend_aligned_bullish if enhanced_bullish else trend_aligned_bearish
            )
            
            touches = detect_level_touches(df, pivot_level['value'])
            touch_info = f" (Touches: {len(touches)})" if touches else ""
            
            message = f"""
🌟 FIFTH SIGNAL - ENHANCED PRIMARY {signal_type} 🌟

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M {pivot_level['type'].title()} at ₹{pivot_level['value']:.2f}{touch_info}
🎯 ATM: {row['Strike']}

⭐ Signal Strength: {signal_strength}/10
📈 Market Trend: {market_trend.title()}
🔊 Volume: {'Strong' if volume_ok else 'Weak'}
🎯 Confluence: {'Yes' if confluence_ok else 'No'}
💪 Options Flow: {'Strong' if options_strong else 'Weak'}
📊 Volatility: {volatility.title()}

All Conditions Met: Pivot + Bias + Filters
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success(f"🌟 ENHANCED FIFTH {signal_type} signal sent! Strength: {signal_strength}/10")
    
    # SIXTH SIGNAL - CONFLUENCE + STRONG OPTIONS FLOW (NO PIVOT REQUIRED)
    strong_confluence_bullish = (
        bias_aligned_bullish and
        trend_aligned_bullish and
        volume_ok and
        options_strong and
        volatility_normal and
        time_ok and
        put_dominance  # Strong PUT dominance suggesting bullish move
    )
    
    strong_confluence_bearish = (
        bias_aligned_bearish and
        trend_aligned_bearish and
        volume_ok and
        options_strong and
        volatility_normal and
        time_ok and
        call_dominance  # Strong CALL dominance suggesting bearish move
    )
    
    if strong_confluence_bullish or strong_confluence_bearish:
        signal_type = "CALL" if strong_confluence_bullish else "PUT"
        dominance_ratio = pe_chg_oi / ce_chg_oi if strong_confluence_bullish and ce_chg_oi > 0 else ce_chg_oi / pe_chg_oi if ce_chg_oi > 0 else 0
        
        # Calculate signal strength
        signal_strength = calculate_signal_strength(
            volume_ok, confluence_ok, options_strong, 
            volatility_normal, time_ok, 
            trend_aligned_bullish if strong_confluence_bullish else trend_aligned_bearish
        )
        
        message = f"""
🚀 SIXTH SIGNAL - CONFLUENCE + FLOW {signal_type} 🚀

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

⭐ Signal Strength: {signal_strength}/10
📈 Market Trend: {market_trend.title()}
🔥 OI Dominance: {'PUT' if strong_confluence_bullish else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
🔊 Volume: Strong
💪 Options Flow: Strong
📊 Volatility: Normal

All Premium Filters Passed
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}
ChgOI: CE {ce_chg_oi:,} | PE {pe_chg_oi:,}
PCR: {row['PCR']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success(f"🚀 CONFLUENCE SIXTH {signal_type} signal sent! Strength: {signal_strength}/10")

def main():
    st.title("📈 Nifty Trading Analyzer")
    
    # Show market status
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"⚠️ Market is closed. Current time: {current_time.strftime('%H:%M:%S IST')}")
        st.info("Market hours: Monday-Friday, 9:00 AM to 3:45 PM IST")
    
    st.sidebar.header("Settings")
    interval = st.sidebar.selectbox("Timeframe", ["1", "3", "5", "10", "15"], index=2)
    proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
    enable_signals = st.sidebar.checkbox("Enable Signals", value=True)
    
    api = DhanAPI()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chart")
        
        data = api.get_intraday_data(interval)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        ltp_data = api.get_ltp_data()
        current_price = None
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
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
        
        if not df.empty:
            fig = create_chart(df, f"Nifty {interval}min")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show nearby pivot levels info
            if current_price:
                nearby_levels = get_nearby_pivot_levels(df, current_price, proximity)
                if nearby_levels:
                    st.info(f"📍 Nearby Levels: {len(nearby_levels)} confirmed pivot levels within {proximity} points")
                    for i, level in enumerate(nearby_levels[:3]):  # Show top 3
                        status = "✓" if level.get('confirmed', True) else "?"
                        st.caption(f"{i+1}. {level['timeframe']}M {level['type'].title()} {status}: ₹{level['value']:.1f} (Distance: {level['distance']:.1f})")
                
                # Show enhanced signal status
                if len(df) > 20:
                    market_trend = get_market_trend(df)
                    volume_ok = check_volume_confirmation(df)
                    volatility = get_market_volatility(df)
                    time_ok = is_good_signal_time()
                    
                    st.subheader("Signal Environment")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Trend", market_trend.title())
                    with col_b:
                        st.metric("Volume", "Strong" if volume_ok else "Weak")
                    with col_c:
                        st.metric("Volatility", volatility.title())
                    with col_d:
                        st.metric("Time", "Good" if time_ok else "Poor")
        else:
            st.error("No chart data available")
    
    with col2:
        st.header("Options")
        
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
    
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time}")
    
    if st.sidebar.button("Test Telegram"):
        send_telegram("🔔 Test message from Nifty Analyzer")
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()