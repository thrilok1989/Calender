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

def get_pivots(df, timeframe="5", length=6, current_price=None, false_breakout_threshold=15, true_breakout_threshold=30):
    """
    Calculate pivot points with reasonable filtering and breakout status
    """
    if df.empty:
        return []
    
    rule_map = {"3": "3min", "5": "5min", "10": "10min", "15": "15min", "30": "30min"}
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
        
        # Get pivot highs
        for timestamp, value in resampled['high'][resampled['high'] == max_vals].items():
            pivots.append({
                'type': 'high', 
                'timeframe': timeframe, 
                'timestamp': timestamp, 
                'value': value,
                'age_hours': (datetime.now(pytz.timezone('Asia/Kolkata')) - timestamp).total_seconds() / 3600,
                'status': 'active'  # Will be updated based on breakout analysis
            })
        
        # Get pivot lows  
        for timestamp, value in resampled['low'][resampled['low'] == min_vals].items():
            pivots.append({
                'type': 'low', 
                'timeframe': timeframe, 
                'timestamp': timestamp, 
                'value': value,
                'age_hours': (datetime.now(pytz.timezone('Asia/Kolkata')) - timestamp).total_seconds() / 3600,
                'status': 'active'  # Will be updated based on breakout analysis
            })
        
        # Filter recent pivots (within last 12 hours - more lenient)
        recent_pivots = [p for p in pivots if p['age_hours'] <= 12]
        
        # Less restrictive filtering
        if current_price and recent_pivots:
            filtered_pivots = []
            for pivot in recent_pivots:
                # Keep pivots within 5% of current price (more lenient)
                price_diff_pct = abs(pivot['value'] - current_price) / current_price * 100
                if price_diff_pct <= 5.0:  # Within 5% of current price
                    # Check if this pivot is too close to existing ones (reduced to 5 points)
                    is_duplicate = False
                    for existing in filtered_pivots:
                        if abs(pivot['value'] - existing['value']) <= 5:  # Within 5 points
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        filtered_pivots.append(pivot)
            recent_pivots = filtered_pivots
        
        # Analyze breakout status if we have current price and recent data
        if current_price and not df.empty:
            recent_pivots = analyze_pivot_breakouts(recent_pivots, df, current_price, false_breakout_threshold, true_breakout_threshold)
        
        # Sort by relevance (closer to current price first) then by time
        if current_price:
            recent_pivots.sort(key=lambda x: abs(x['value'] - current_price))
        else:
            recent_pivots.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return recent_pivots
    except Exception as e:
        print(f"Pivot calculation error: {e}")
        return []

def get_pivots_with_thresholds(df, timeframe="5", length=6, current_price=None, false_threshold=15, true_threshold=30):
    """
    Wrapper function to get pivots with custom thresholds
    """
    return get_pivots(df, timeframe, length, current_price, false_threshold, true_threshold)

def create_chart_with_thresholds(df, title, current_price, max_pivots_per_timeframe=3, pivot_length=6, false_threshold=15, true_threshold=30):
    """
    Create chart with custom breakout thresholds
    """
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)
    
    if len(df) > 50:
        # Use 5min and 15min timeframes for better coverage
        timeframes = ["5", "15"]
        base_colors = {"5": "#ff9900", "15": "#00ccff"}
        
        # Status colors for different breakout states
        status_colors = {
            'active': '',  # Use base color
            'testing': '#ffff00',  # Yellow for testing
            'false_breakout': '#ff6600',  # Orange for false breakout  
            'broken': '#ff0000'  # Red for broken pivots
        }
        
        x_start, x_end = df['datetime'].min(), df['datetime'].max()
        
        for tf in timeframes:
            pivots = get_pivots(df, tf, length=pivot_length, current_price=current_price, 
                              false_breakout_threshold=false_threshold, true_breakout_threshold=true_threshold)
            
            # Limit number of pivots per timeframe
            pivots_to_show = pivots[:max_pivots_per_timeframe]
            
            for pivot in pivots_to_show:
                # Determine color based on status
                status = pivot.get('status', 'active')
                if status == 'active':
                    color = base_colors[tf]
                    dash = "dash"
                    width = 2
                elif status == 'testing':
                    color = status_colors['testing']
                    dash = "dot"
                    width = 2
                elif status == 'false_breakout':
                    color = status_colors['false_breakout']
                    dash = "dashdot"
                    width = 3
                elif status == 'broken':
                    color = status_colors['broken']
                    dash = "solid"
                    width = 1
                else:
                    color = base_colors[tf]
                    dash = "dash"
                    width = 2
                
                # Add pivot line
                fig.add_shape(
                    type="line", 
                    x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=width, dash=dash), 
                    row=1, col=1
                )
                
                # Create label with status info
                status_symbol = {
                    'active': '✓',
                    'testing': '⚠️', 
                    'false_breakout': '↩️',
                    'broken': '❌'
                }.get(status, '✓')
                
                label_text = f"{pivot['value']:.0f} ({tf}m) {status_symbol}"
                
                # Add pivot label with timeframe and status
                fig.add_annotation(
                    x=x_end,
                    y=pivot['value'],
                    text=label_text,
                    showarrow=False,
                    font=dict(color=color, size=9),
                    bgcolor="rgba(0,0,0,0.7)",
                    row=1, col=1
                )
    
    fig.update_layout(title=title, template='plotly_dark', height=600,
                     xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def check_signals_with_breakouts(df, option_data, current_price, proximity=5, pivots=None, enable_breakout_alerts=True):
    """
    Enhanced signal checking that includes breakout analysis
    """
    # Run original signals first
    check_signals(df, option_data, current_price, proximity)
    
    # Add breakout-specific alerts
    if enable_breakout_alerts and pivots and is_market_hours():
        check_pivot_breakout_signals(pivots, current_price)

def check_pivot_breakout_signals(pivots, current_price):
    """
    Send Telegram alerts for pivot breakouts
    """
    if not pivots:
        return
    
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime('%H:%M:%S IST')
    
    for pivot in pivots:
        status = pivot.get('status', 'active')
        pivot_value = pivot['value']
        timeframe = pivot['timeframe']
        pivot_type = pivot['type']
        
        # Send alerts for significant events
        if status == 'broken':
            breakout_points = pivot.get('breakout_points', 0)
            direction = "UPWARD" if pivot_type == 'high' else "DOWNWARD"
            level_type = "Resistance" if pivot_type == 'high' else "Support"
            
            message = f"""
🚨 PIVOT BREAKOUT ALERT! 🚨

{level_type} BROKEN: ₹{pivot_value:.0f} ({timeframe}M)
📍 Current Price: ₹{current_price:.2f}
🔥 Breakout: {breakout_points:.1f} points beyond pivot
📈 Direction: {direction} MOMENTUM

⚡ Action: Watch for continuation or retest
🕐 Time: {current_time}
"""
            send_telegram(message)
            st.success(f"🚨 {level_type} breakout alert sent!")
            
        elif status == 'false_breakout':
            max_breach = pivot.get('max_breach', 0)
            level_type = "Resistance" if pivot_type == 'high' else "Support"
            
            message = f"""
🔄 FALSE BREAKOUT ALERT! 🔄

{level_type}: ₹{pivot_value:.0f} ({timeframe}M)
📍 Current Price: ₹{current_price:.2f}
↩️ False Break: Breached by {max_breach:.1f} points but PULLED BACK

✅ Level Still Valid - Watch for Bounce/Rejection
🕐 Time: {current_time}
"""
            send_telegram(message)
            st.info(f"🔄 False breakout alert sent!")
            
        elif status == 'testing':
            breakout_points = pivot.get('breakout_points', 0)
            level_type = "Resistance" if pivot_type == 'high' else "Support"
            direction = "above" if pivot_type == 'high' else "below"
            
            message = f"""
⚠️ PIVOT TESTING ALERT! ⚠️

{level_type}: ₹{pivot_value:.0f} ({timeframe}M)
📍 Current Price: ₹{current_price:.2f}
🎯 Testing: {breakout_points:.1f} points {direction} pivot

⚡ CRITICAL LEVEL - Watch for breakout or rejection!
🕐 Time: {current_time}
"""
            send_telegram(message)
            st.warning(f"⚠️ Pivot testing alert sent!")
    """
    Check for breakout alerts and display them
    """
    if not pivots:
        return
        
    # Check for recent breakouts
    for pivot in pivots:
        status = pivot.get('status', 'active')
        pivot_value = pivot['value']
        timeframe = pivot['timeframe']
        pivot_type = pivot['type']
        
        if status == 'false_breakout':
            st.warning(f"🔄 **False Breakout Detected!** {pivot_type.title()} pivot at {pivot_value:.0f} ({timeframe}m) - Price pulled back after breach")
            
        elif status == 'broken':
            breakout_points = pivot.get('breakout_points', 0)
            if pivot_type == 'high':
                st.error(f"📈 **Resistance BROKEN!** {pivot_value:.0f} ({timeframe}m) breached by {breakout_points:.1f} points - Potential upward momentum")
            else:
                st.error(f"📉 **Support BROKEN!** {pivot_value:.0f} ({timeframe}m) breached by {breakout_points:.1f} points - Potential downward pressure")
                
        elif status == 'testing':
            breakout_points = pivot.get('breakout_points', 0)
            if pivot_type == 'high':
                st.info(f"⚡ **Resistance Testing** {pivot_value:.0f} ({timeframe}m) - Price {breakout_points:.1f} points above, watch for breakout or rejection")
            else:
                st.info(f"⚡ **Support Testing** {pivot_value:.0f} ({timeframe}m) - Price {breakout_points:.1f} points below, watch for breakdown or bounce")
    """
    Analyze pivot breakout status:
    - active: Pivot is holding, price hasn't broken it significantly
    - false_breakout: Price broke but came back (pullback happened)
    - broken: True breakout, price moved significantly beyond pivot
    """
    if df.empty or not pivots:
        return pivots
    
    # Get recent price action (last 10 candles for analysis)
    recent_df = df.tail(20)
    
    for pivot in pivots:
        pivot_value = pivot['value']
        pivot_type = pivot['type']
        
        # Check current distance from pivot
        current_distance = current_price - pivot_value
        
        if pivot_type == 'high':
            # For resistance levels (pivot highs)
            if current_distance > true_breakout_threshold:
                pivot['status'] = 'broken'
                pivot['breakout_points'] = current_distance
            elif current_distance > false_breakout_threshold:
                # Check if it's a false breakout (price came back)
                max_breach = (recent_df['high'] - pivot_value).max()
                if max_breach > false_breakout_threshold and current_distance < false_breakout_threshold:
                    pivot['status'] = 'false_breakout'
                    pivot['max_breach'] = max_breach
                else:
                    pivot['status'] = 'testing'
                    pivot['breakout_points'] = current_distance
            else:
                pivot['status'] = 'active'
                
        else:  # pivot_type == 'low'
            # For support levels (pivot lows)
            current_distance = pivot_value - current_price  # Flip for support
            if current_distance > true_breakout_threshold:
                pivot['status'] = 'broken'
                pivot['breakout_points'] = current_distance
            elif current_distance > false_breakout_threshold:
                # Check if it's a false breakout (price came back)
                max_breach = (pivot_value - recent_df['low']).max()
                if max_breach > false_breakout_threshold and current_distance < false_breakout_threshold:
                    pivot['status'] = 'false_breakout'
                    pivot['max_breach'] = max_breach
                else:
                    pivot['status'] = 'testing'
                    pivot['breakout_points'] = current_distance
            else:
                pivot['status'] = 'active'
    
    return pivots

def create_chart(df, title, current_price, max_pivots_per_timeframe=3):
    """
    Chart with controlled number of pivot levels and breakout status colors
    """
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)
    
    if len(df) > 50:
        # Use 5min and 15min timeframes for better coverage
        timeframes = ["5", "15"]
        base_colors = {"5": "#ff9900", "15": "#00ccff"}
        
        # Status colors for different breakout states
        status_colors = {
            'active': '',  # Use base color
            'testing': '#ffff00',  # Yellow for testing
            'false_breakout': '#ff6600',  # Orange for false breakout  
            'broken': '#ff0000'  # Red for broken pivots
        }
        
        x_start, x_end = df['datetime'].min(), df['datetime'].max()
        
        for tf in timeframes:
            pivots = get_pivots(df, tf, length=6, current_price=current_price)
            
            # Limit number of pivots per timeframe
            pivots_to_show = pivots[:max_pivots_per_timeframe]
            
            for pivot in pivots_to_show:
                # Determine color based on status
                status = pivot.get('status', 'active')
                if status == 'active':
                    color = base_colors[tf]
                    dash = "dash"
                    width = 2
                elif status == 'testing':
                    color = status_colors['testing']
                    dash = "dot"
                    width = 2
                elif status == 'false_breakout':
                    color = status_colors['false_breakout']
                    dash = "dashdot"
                    width = 3
                elif status == 'broken':
                    color = status_colors['broken']
                    dash = "solid"
                    width = 1
                else:
                    color = base_colors[tf]
                    dash = "dash"
                    width = 2
                
                # Add pivot line
                fig.add_shape(
                    type="line", 
                    x0=x_start, x1=x_end,
                    y0=pivot['value'], y1=pivot['value'],
                    line=dict(color=color, width=width, dash=dash), 
                    row=1, col=1
                )
                
                # Create label with status info
                status_symbol = {
                    'active': '✓',
                    'testing': '⚠️', 
                    'false_breakout': '↩️',
                    'broken': '❌'
                }.get(status, '✓')
                
                label_text = f"{pivot['value']:.0f} ({tf}m) {status_symbol}"
                
                # Add pivot label with timeframe and status
                fig.add_annotation(
                    x=x_end,
                    y=pivot['value'],
                    text=label_text,
                    showarrow=False,
                    font=dict(color=color, size=9),
                    bgcolor="rgba(0,0,0,0.7)",
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
    
    # PRIMARY SIGNAL - Check both 5min and 15min pivots with user thresholds
    pivots_5m = get_pivots(df, "5", length=6, current_price=current_price, 
                          false_breakout_threshold=15, true_breakout_threshold=30)
    pivots_15m = get_pivots(df, "15", length=6, current_price=current_price,
                           false_breakout_threshold=15, true_breakout_threshold=30)
    all_pivots = pivots_5m + pivots_15m
    
    near_pivot = False
    pivot_level = None
    
    for pivot in all_pivots:
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
📌 Pivot: {pivot_level['timeframe']}M at ₹{pivot_level['value']:.2f}
🎯 ATM: {row['Strike']}

Conditions: {row['Level']}, All Bias Aligned
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
    
    # New pivot controls
    st.sidebar.subheader("Pivot Controls")
    max_pivots = st.sidebar.slider("Max Pivots to Show", 1, 5, 2)
    pivot_sensitivity = st.sidebar.selectbox("Pivot Sensitivity", ["Low", "Medium", "High"], index=1)
    
    # Breakout detection controls
    st.sidebar.subheader("Breakout Detection")
    false_breakout_threshold = st.sidebar.slider("False Breakout Threshold (points)", 10, 30, 15)
    true_breakout_threshold = st.sidebar.slider("True Breakout Threshold (points)", 20, 50, 30)
    enable_breakout_alerts = st.sidebar.checkbox("Enable Breakout Alerts", value=True)
    
    # Map sensitivity to length parameter
    sensitivity_map = {"High": 4, "Medium": 6, "Low": 8}
    pivot_length = sensitivity_map[pivot_sensitivity]
    
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
            # Use improved chart function with current price and thresholds
            # Update pivot calculations with user-defined thresholds
            fig = create_chart_with_thresholds(df, f"Nifty {interval}min", current_price, max_pivots, pivot_length, false_breakout_threshold, true_breakout_threshold)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show pivot info with breakout status
            if current_price:
                pivots_5m = get_pivots_with_thresholds(df, "5", length=pivot_length, current_price=current_price, 
                                                     false_threshold=false_breakout_threshold, true_threshold=true_breakout_threshold)
                pivots_15m = get_pivots_with_thresholds(df, "15", length=pivot_length, current_price=current_price,
                                                      false_threshold=false_breakout_threshold, true_threshold=true_breakout_threshold)
                all_pivots = (pivots_5m + pivots_15m)[:5]  # Combine and limit to 5
                
                if all_pivots:
                    st.subheader("Active Pivot Levels")
                    
                    # Create enhanced pivot table with breakout status
                    pivot_data = []
                    for p in all_pivots:
                        status_emoji = {
                            'active': '✅',
                            'testing': '⚠️', 
                            'false_breakout': '↩️',
                            'broken': '❌'
                        }.get(p.get('status', 'active'), '✅')
                        
                        distance = current_price - p['value']
                        
                        row_data = {
                            'Level': f"{p['value']:.0f}",
                            'TF': f"{p['timeframe']}m",
                            'Type': p['type'].title(),
                            'Status': f"{status_emoji} {p.get('status', 'active').title()}",
                            'Distance': f"{distance:+.1f}",
                            'Age': f"{p['age_hours']:.1f}h"
                        }
                        
                        # Add breakout info if available
                        if 'breakout_points' in p:
                            row_data['Breakout'] = f"{p['breakout_points']:.1f}"
                        elif 'max_breach' in p:
                            row_data['Max Breach'] = f"{p['max_breach']:.1f}"
                            
                        pivot_data.append(row_data)
                    
                    pivot_df = pd.DataFrame(pivot_data)
                    st.dataframe(pivot_df, use_container_width=True)
                    
                    # Show breakout statistics
                    status_counts = {}
                    for p in all_pivots:
                        status = p.get('status', 'active')
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    # Create metrics row for breakout summary
                    if status_counts:
                        cols = st.columns(len(status_counts))
                        for i, (status, count) in enumerate(status_counts.items()):
                            emoji_map = {'active': '✅', 'testing': '⚠️', 'false_breakout': '↩️', 'broken': '❌'}
                            with cols[i]:
                                st.metric(f"{emoji_map.get(status, '•')} {status.title()}", count)
                    
                    # Show breakout legend
                    st.caption("""
                    **Status Legend:** ✅ Active | ⚠️ Testing | ↩️ False Breakout | ❌ Broken  
                    **Thresholds:** False BO: {false_breakout_threshold} pts | True BO: {true_breakout_threshold} pts
                    """.format(false_breakout_threshold=false_breakout_threshold, true_breakout_threshold=true_breakout_threshold))
                    
                    # Check for breakout alerts
                    if enable_breakout_alerts:
                        check_breakout_alerts(all_pivots, current_price)
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
                    # Get pivots with current breakout thresholds for signals
                    all_pivots_for_signals = []
                    if current_price:
                        pivots_5m_signals = get_pivots_with_thresholds(df, "5", length=pivot_length, current_price=current_price,
                                                                     false_threshold=false_breakout_threshold, true_threshold=true_breakout_threshold)
                        pivots_15m_signals = get_pivots_with_thresholds(df, "15", length=pivot_length, current_price=current_price,
                                                                      false_threshold=false_breakout_threshold, true_threshold=true_breakout_threshold)
                        all_pivots_for_signals = pivots_5m_signals + pivots_15m_signals
                    
                    check_signals_with_breakouts(df, option_summary, underlying_price, proximity, all_pivots_for_signals, enable_breakout_alerts)
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
