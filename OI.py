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
import websocket
import json
import struct
import threading
import time

# Page config
st.set_page_config(page_title="Advanced Nifty Analyzer", page_icon="📈", layout="wide")

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
    
    def get_market_quote(self, instruments_dict):
        """Get full market quote data including depth"""
        url = "https://api.dhan.co/v2/marketfeed/quote"
        try:
            response = requests.post(url, headers=self.headers, json=instruments_dict)
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

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['close'].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_vwap_bands(df):
    """Calculate VWAP with standard deviation bands"""
    if df.empty:
        return df
    
    # Calculate typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate cumulative values
    df['cum_vol'] = df['volume'].cumsum()
    df['cum_tp_vol'] = (df['typical_price'] * df['volume']).cumsum()
    
    # Calculate VWAP
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    
    # Calculate squared deviations for standard deviation
    df['sq_dev'] = (df['typical_price'] - df['vwap']) ** 2
    df['cum_sq_dev_vol'] = (df['sq_dev'] * df['volume']).cumsum()
    
    # Calculate VWAP standard deviation
    df['vwap_std'] = np.sqrt(df['cum_sq_dev_vol'] / df['cum_vol'])
    
    # Calculate bands
    df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
    df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
    df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
    df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']
    
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    if df.empty or len(df) < period:
        return df
    
    # Calculate True Range
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['l_pc'] = abs(df['low'] - df['close'].shift(1))
    
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    return df

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_implied_volatility(option_price, spot_price, strike_price, time_to_expiry, option_type='CE', risk_free_rate=0.07):
    """Calculate IV using Newton-Raphson method"""
    try:
        if time_to_expiry <= 0 or option_price <= 0:
            return 0
        
        # Initial guess for volatility
        iv = 0.3
        
        for _ in range(50):  # Max iterations
            if option_type == 'CE':
                theoretical_price = black_scholes_call(spot_price, strike_price, time_to_expiry, risk_free_rate, iv)
            else:
                theoretical_price = black_scholes_put(spot_price, strike_price, time_to_expiry, risk_free_rate, iv)
            
            d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * iv**2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
            
            if abs(theoretical_price - option_price) < 0.001 or vega == 0:
                break
                
            iv = iv - (theoretical_price - option_price) / vega
            iv = max(0.001, min(5.0, iv))  # Clamp between 0.1% and 500%
            
        return iv
    except:
        return 0

def calculate_gamma(spot_price, strike_price, time_to_expiry, iv, risk_free_rate=0.07):
    """Calculate option gamma"""
    try:
        if time_to_expiry <= 0 or iv <= 0:
            return 0
        
        d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * iv**2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
        gamma = norm.pdf(d1) / (spot_price * iv * np.sqrt(time_to_expiry))
        return gamma
    except:
        return 0

def analyze_volume_spread(df_options, spot_price):
    """Analyze volume spread and cumulative delta with correct bias logic"""
    results = []
    
    for _, row in df_options.iterrows():
        ce_vol = row.get('volume_CE', 0)
        pe_vol = row.get('volume_PE', 0)
        
        ce_bid_qty = row.get('top_bid_quantity_CE', 0)
        ce_ask_qty = row.get('top_ask_quantity_CE', 0)
        pe_bid_qty = row.get('top_bid_quantity_PE', 0)
        pe_ask_qty = row.get('top_ask_quantity_PE', 0)
        
        # Approximate buy/sell volume based on bid/ask imbalance
        ce_buy_vol = ce_vol * 0.6 if (ce_bid_qty > ce_ask_qty) else ce_vol * 0.4
        ce_sell_vol = ce_vol - ce_buy_vol
        
        pe_buy_vol = pe_vol * 0.6 if (pe_bid_qty > pe_ask_qty) else pe_vol * 0.4
        pe_sell_vol = pe_vol - pe_buy_vol
        
        ce_delta = ce_buy_vol - ce_sell_vol
        pe_delta = pe_buy_vol - pe_sell_vol
        
        # CORRECTED BIAS LOGIC:
        # CE: Buy Vol > Sell Vol = Bullish, Sell Vol > Buy Vol = Bearish
        ce_delta_bias = "Bullish" if ce_delta > 0 else "Bearish" if ce_delta < 0 else "Neutral"
        
        # PE: Sell Vol > Buy Vol = Bullish (selling puts = bullish), Buy Vol > Sell Vol = Bearish (buying puts = bearish)
        pe_delta_bias = "Bullish" if pe_delta < 0 else "Bearish" if pe_delta > 0 else "Neutral"
        
        results.append({
            'Strike': row['strikePrice'],
            'CE_Buy_Vol': round(ce_buy_vol, 0),
            'CE_Sell_Vol': round(ce_sell_vol, 0),
            'PE_Buy_Vol': round(pe_buy_vol, 0),
            'PE_Sell_Vol': round(pe_sell_vol, 0),
            'CE_Cumulative_Delta': round(ce_delta, 0),
            'PE_Cumulative_Delta': round(pe_delta, 0),
            'CE_Delta_Bias': ce_delta_bias,
            'PE_Delta_Bias': pe_delta_bias
        })
    
    return pd.DataFrame(results)

def get_pivots(df, timeframe="5", length=4):
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

def calculate_volume_profile(df, bins=20):
    """Calculate Volume Profile for price action analysis"""
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    
    profiles = []
    
    # Calculate for each candle
    for idx, row in df.iterrows():
        high = row['high']
        low = row['low']
        volume = row['volume']
        close = row['close']
        
        if high == low:  # Avoid division by zero
            continue
            
        # Create price bins
        bins_edges = np.linspace(low, high, bins + 1)
        bins_center = (bins_edges[:-1] + bins_edges[1:]) / 2
        vol_bins = np.zeros(bins)
        
        # Distribute volume across bins (simplified approach)
        # In reality, you'd need tick data for accurate distribution
        close_bin = np.digitize(close, bins_edges) - 1
        close_bin = max(0, min(bins - 1, close_bin))
        
        # Distribute volume: more weight to close price bin
        for i in range(bins):
            if i == close_bin:
                vol_bins[i] = volume * 0.4  # 40% to close bin
            else:
                # Distribute remaining volume based on distance from close
                distance = abs(bins_center[i] - close)
                max_distance = max(abs(bins_center - close))
                if max_distance > 0:
                    weight = 1 - (distance / max_distance)
                    vol_bins[i] = volume * 0.6 * weight / (bins - 1)
        
        # Point of Control (POC) - price level with highest volume
        poc_idx = np.argmax(vol_bins)
        poc = bins_center[poc_idx]
        
        profiles.append({
            'datetime': row['datetime'],
            'high': high,
            'low': low,
            'close': close,
            'poc': poc,
            'vol_bins': vol_bins,
            'bins_center': bins_center,
            'max_vol': np.max(vol_bins)
        })
    
    return pd.DataFrame(profiles)

def create_enhanced_chart(df, title):
    if df.empty:
        return go.Figure()
    
    # Calculate indicators
    df = calculate_vwap_bands(df)
    df = calculate_atr(df)
    df['rsi'] = calculate_rsi(df)
    
    # Calculate Volume Profile
    volume_profile_df = calculate_volume_profile(df)
    
    # Create subplots with 5 rows (added Volume Profile)
    fig = make_subplots(
        rows=5, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=('Price with VWAP Bands', 'Volume Profile', 'Volume', 'RSI', 'ATR')
    )
    
    # Price chart with VWAP bands
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # VWAP and bands
    if 'vwap' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['vwap'], name='VWAP',
            line=dict(color='#ffff00', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['vwap_upper_1'], name='VWAP +1σ',
            line=dict(color='#ff9900', width=1, dash='dash'), opacity=0.7
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['vwap_lower_1'], name='VWAP -1σ',
            line=dict(color='#ff9900', width=1, dash='dash'), opacity=0.7
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['vwap_upper_2'], name='VWAP +2σ',
            line=dict(color='#ff0000', width=1, dash='dot'), opacity=0.5
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['vwap_lower_2'], name='VWAP -2σ',
            line=dict(color='#ff0000', width=1, dash='dot'), opacity=0.5
        ), row=1, col=1)
    
    # Add pivot levels
    if len(df) > 50:
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", '#4444ff']
        
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            x_start, x_end = df['datetime'].min(), df['datetime'].max()
            
            for pivot in pivots[-5:]:
                fig.add_shape(type="line", x0=x_start, x1=x_end,
                            y0=pivot['value'], y1=pivot['value'],
                            line=dict(color=color, width=1, dash="dash"), row=1, col=1)
    
    # Volume Profile
    if not volume_profile_df.empty:
        # Add POC lines (Point of Control)
        fig.add_trace(go.Scatter(
            x=volume_profile_df['datetime'], 
            y=volume_profile_df['poc'], 
            mode='lines',
            name='POC (Point of Control)',
            line=dict(color='#0066ff', width=2, dash='dash')
        ), row=2, col=1)
        
        # Add volume profile bars for recent periods
        for idx, row in volume_profile_df.tail(20).iterrows():  # Last 20 candles
            if row['max_vol'] > 0:
                # Normalize volume bins for display
                normalized_vol = row['vol_bins'] / row['max_vol'] * 0.8  # Scale factor
                
                for i, (price, vol_norm) in enumerate(zip(row['bins_center'], normalized_vol)):
                    if vol_norm > 0.1:  # Only show significant volume
                        fig.add_trace(go.Bar(
                            x=[row['datetime']],
                            y=[vol_norm],
                            base=[price - (row['bins_center'][1] - row['bins_center'][0])/2],
                            width=[pd.Timedelta(minutes=int(interval)) * 0.8],
                            marker_color='lightblue',
                            opacity=0.6,
                            showlegend=False
                        ), row=2, col=1)
    
    # Regular Volume chart
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=3, col=1)
    
    # RSI chart
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['rsi'], name='RSI',
        line=dict(color='#ff9900', width=2)
    ), row=4, col=1)
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=4, col=1)
    
    # ATR chart
    if 'atr' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['atr'], name='ATR',
            line=dict(color='#00ffff', width=2)
        ), row=5, col=1)
    
    fig.update_layout(title=title, template='plotly_dark', height=1200,
                     xaxis_rangeslider_visible=False, showlegend=False)
    
    # Update y-axis ranges
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)
    fig.update_yaxes(title_text="ATR", row=5, col=1)
    fig.update_yaxes(title_text="Volume Profile", row=2, col=1)
    
    return fig

def analyze_advanced_options(expiry, spot_price):
    """Enhanced options analysis with all new features"""
    option_data = get_option_chain(expiry)
    if not option_data or 'data' not in option_data:
        return None, None, None, None, None
    
    data = option_data['data']
    underlying = data['last_price']
    oc_data = data['oc']
    
    calls, puts = [], []
    for strike, strike_data in oc_data.items():
        strike_price = float(strike)
        if 'ce' in strike_data:
            ce_data = strike_data['ce'].copy()
            ce_data['strikePrice'] = strike_price
            calls.append(ce_data)
        if 'pe' in strike_data:
            pe_data = strike_data['pe'].copy()
            pe_data['strikePrice'] = strike_price
            puts.append(pe_data)
    
    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    
    if df_ce.empty or df_pe.empty:
        return None, None, None, None, None
    
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
    
    # Filter ATM ±2 strikes (100 points range for Nifty)
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df_filtered = df[abs(df['strikePrice'] - atm_strike) <= 100]
    
    if df_filtered.empty:
        return None, None, None, None, None
    
    # Calculate time to expiry
    try:
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        expiry_datetime = expiry_date.replace(hour=15, minute=30)
        current_time = datetime.now()
        time_to_expiry = max((expiry_datetime - current_time).total_seconds() / (365.25 * 24 * 3600), 0.001)
    except:
        time_to_expiry = 0.027  # Default to ~10 days
    
    # Calculate IV and Gamma for each strike
    iv_results = []
    gamma_results = []
    
    for _, row in df_filtered.iterrows():
        strike = row['strikePrice']
        
        # Calculate IV
        ce_price = row.get('last_price_CE', 0)
        pe_price = row.get('last_price_PE', 0)
        
        ce_iv = calculate_implied_volatility(ce_price, underlying, strike, time_to_expiry, 'CE')
        pe_iv = calculate_implied_volatility(pe_price, underlying, strike, time_to_expiry, 'PE')
        
        # Calculate Gamma
        ce_gamma = calculate_gamma(underlying, strike, time_to_expiry, ce_iv)
        pe_gamma = calculate_gamma(underlying, strike, time_to_expiry, pe_iv)
        
        # IV Bias (comparing to historical average of ~15-25%)
        ce_iv_bias = "Expensive" if ce_iv > 0.25 else "Cheap" if ce_iv < 0.15 else "Fair"
        pe_iv_bias = "Expensive" if pe_iv > 0.25 else "Cheap" if pe_iv < 0.15 else "Fair"
        
        # Gamma Exposure
        ce_oi = row.get('oi_CE', 0)
        pe_oi = row.get('oi_PE', 0)
        
        ce_gex = ce_gamma * ce_oi * underlying * 0.01  # GEX in points
        pe_gex = pe_gamma * pe_oi * underlying * 0.01
        net_gex = ce_gex - pe_gex  # Net gamma exposure
        
        gamma_bias = "High Volatility" if abs(net_gex) > 2000 else "Pinned" if abs(net_gex) < 500 else "Moderate"
        
        iv_results.append({
            'Strike': strike,
            'CE_IV': round(ce_iv * 100, 2),
            'PE_IV': round(pe_iv * 100, 2),
            'CE_IV_Bias': ce_iv_bias,
            'PE_IV_Bias': pe_iv_bias
        })
        
        gamma_results.append({
            'Strike': strike,
            'CE_Gamma': round(ce_gamma, 4),
            'PE_Gamma': round(pe_gamma, 4),
            'CE_GEX': round(ce_gex, 0),
            'PE_GEX': round(pe_gex, 0),
            'Net_GEX': round(net_gex, 0),
            'Gamma_Bias': gamma_bias
        })
    
    # Basic options analysis with enhanced bias columns
    df_filtered['Zone'] = df_filtered['strikePrice'].apply(
        lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM'
    )
    
    bias_results = []
    for i, row in df_filtered.iterrows():
        strike = row['strikePrice']
        
        # Basic OI and volume data
        prev_oi_ce = row.get('previous_oi_CE', 0)
        prev_oi_pe = row.get('previous_oi_PE', 0)
        curr_oi_ce = row.get('oi_CE', 0)
        curr_oi_pe = row.get('oi_PE', 0)
        
        chg_oi_ce = curr_oi_ce - prev_oi_ce
        chg_oi_pe = curr_oi_pe - prev_oi_pe
        
        chg_oi_bias = "Bullish" if chg_oi_ce < chg_oi_pe else "Bearish"
        volume_bias = "Bullish" if row.get('volume_CE', 0) < row.get('volume_PE', 0) else "Bearish"
        
        ask_ce = row.get('top_ask_quantity_CE', 0)
        ask_pe = row.get('top_ask_quantity_PE', 0)
        bid_ce = row.get('top_bid_quantity_CE', 0)
        bid_pe = row.get('top_bid_quantity_PE', 0)
        
        ask_bias = "Bearish" if ask_ce > ask_pe else "Bullish"
        bid_bias = "Bullish" if bid_ce > bid_pe else "Bearish"
        
        level = "Support" if curr_oi_pe > 1.12 * curr_oi_ce else "Resistance" if curr_oi_ce > 1.12 * curr_oi_ce else "Neutral"
        
        # Calculate additional bias columns
        
        # 1. CE Delta Bias & PE Delta Bias (from Greeks)
        ce_delta = row.get('greeks', {}).get('delta', 0) if 'greeks' in row else 0
        pe_delta = row.get('greeks', {}).get('delta', 0) if 'greeks' in row else 0
        
        # For CE: positive delta is bullish, for PE: negative delta is bearish (normal)
        ce_delta_bias = "Bullish" if ce_delta > 0.5 else "Bearish" if ce_delta < 0.3 else "Neutral"
        pe_delta_bias = "Bearish" if pe_delta < -0.5 else "Bullish" if pe_delta > -0.3 else "Neutral"
        
        # 2. IV Bias - Compare CE vs PE IV to see which is cheaper (CORRECTED LOGIC)
        ce_price = row.get('last_price_CE', 0)
        pe_price = row.get('last_price_PE', 0)
        
        ce_iv = calculate_implied_volatility(ce_price, underlying, strike, time_to_expiry, 'CE')
        pe_iv = calculate_implied_volatility(pe_price, underlying, strike, time_to_expiry, 'PE')
        
        if ce_iv > 0 and pe_iv > 0:
            iv_diff_pct = abs(ce_iv - pe_iv) / max(ce_iv, pe_iv) * 100
            
            if iv_diff_pct > 5:  # Significant difference (>5%)
                if ce_iv < pe_iv:
                    iv_bias = "CE Cheaper"
                else:
                    iv_bias = "PE Cheaper"
            else:
                iv_bias = "Neutral"  # IVs are similar
        else:
            iv_bias = "N/A"
        
        # 3. Gamma Exposure Bias - Compare CE vs PE Gamma (CORRECTED LOGIC)
        ce_gamma = calculate_gamma(underlying, strike, time_to_expiry, ce_iv)
        pe_gamma = calculate_gamma(underlying, strike, time_to_expiry, pe_iv)
        
        ce_gex = ce_gamma * curr_oi_ce * underlying * 0.01
        pe_gex = pe_gamma * curr_oi_pe * underlying * 0.01
        
        # Compare CE vs PE Gamma Exposure
        if abs(ce_gex) > 0 and abs(pe_gex) > 0:
            gamma_diff_pct = abs(abs(ce_gex) - abs(pe_gex)) / max(abs(ce_gex), abs(pe_gex)) * 100
            
            if gamma_diff_pct > 20:  # Significant difference (>20%)
                if abs(ce_gex) > abs(pe_gex):
                    gamma_bias = "CE Higher Gamma"
                else:
                    gamma_bias = "PE Higher Gamma"
            else:
                gamma_bias = "Neutral"
        else:
            gamma_bias = "N/A"
        
        bias_results.append({
            "Strike": strike,
            "Zone": row['Zone'],
            "Level": level,
            "ChgOI_Bias": chg_oi_bias,
            "Volume_Bias": volume_bias,
            "Ask_Bias": ask_bias,
            "Bid_Bias": bid_bias,
            "CE_Delta_Bias": ce_delta_bias,
            "PE_Delta_Bias": pe_delta_bias,
            "IV_Bias": iv_bias,
            "Gamma_Bias": gamma_bias,
            "PCR": round(curr_oi_pe / curr_oi_ce if curr_oi_ce > 0 else 0, 2),
            "CE_IV": round(ce_iv * 100, 1),
            "PE_IV": round(pe_iv * 100, 1),
            "Net_GEX": round(net_gex, 0),
            "changeinOpenInterest_CE": chg_oi_ce,
            "changeinOpenInterest_PE": chg_oi_pe
        })
    
    # Volume spread analysis
    volume_spread_df = analyze_volume_spread(df_filtered, underlying)
    
    return underlying, pd.DataFrame(bias_results), pd.DataFrame(iv_results), pd.DataFrame(gamma_results), volume_spread_df

def calculate_final_bias_score(df, option_summary, iv_df, gamma_df, volume_spread_df, current_price):
    """Calculate comprehensive bias score"""
    if df.empty or option_summary is None:
        return "Insufficient Data", 0, {}
    
    scores = {}
    
    # RSI Score (-2 to +2)
    df['rsi'] = calculate_rsi(df)
    current_rsi = df['rsi'].iloc[-1] if not df.empty else 50
    if current_rsi > 70:
        rsi_score = -2
    elif current_rsi > 60:
        rsi_score = -1
    elif current_rsi < 30:
        rsi_score = 2
    elif current_rsi < 40:
        rsi_score = 1
    else:
        rsi_score = 0
    scores['RSI'] = rsi_score
    
    # OI Bias Score (-2 to +2) - Updated to include new bias columns
    atm_data = option_summary[option_summary['Zone'] == 'ATM']
    if not atm_data.empty:
        atm_row = atm_data.iloc[0]
        oi_bullish = sum([1 for bias in [atm_row['ChgOI_Bias'], atm_row['Volume_Bias'], 
                         atm_row['Ask_Bias'], atm_row['Bid_Bias'], atm_row['CE_Delta_Bias'], atm_row['PE_Delta_Bias']] 
                         if bias == 'Bullish'])
        oi_score = oi_bullish - 3  # Convert 0-6 to -3 to +3, then clamp to -2 to +2
        oi_score = max(-2, min(2, oi_score))
        scores['OI_Bias'] = oi_score
    else:
        scores['OI_Bias'] = 0
    
    # Volume Delta Score (-1 to +1)
    if volume_spread_df is not None and not volume_spread_df.empty:
        atm_volume = volume_spread_df[volume_spread_df['Strike'].isin(atm_data['Strike'])]
        if not atm_volume.empty:
            avg_ce_delta = atm_volume['CE_Cumulative_Delta'].mean()
            avg_pe_delta = atm_volume['PE_Cumulative_Delta'].mean()
            if avg_ce_delta > abs(avg_pe_delta):
                volume_delta_score = -1
            elif avg_pe_delta > abs(avg_ce_delta):
                volume_delta_score = 1
            else:
                volume_delta_score = 0
        else:
            volume_delta_score = 0
        scores['Volume_Delta'] = volume_delta_score
    else:
        scores['Volume_Delta'] = 0
    
    # IV Score (-1 to +1)
    if iv_df is not None and not iv_df.empty:
        atm_iv = iv_df[iv_df['Strike'].isin(atm_data['Strike'])]
        if not atm_iv.empty:
            avg_ce_iv_bias = 1 if atm_iv['CE_IV_Bias'].iloc[0] == 'Expensive' else -1 if atm_iv['CE_IV_Bias'].iloc[0] == 'Cheap' else 0
            avg_pe_iv_bias = 1 if atm_iv['PE_IV_Bias'].iloc[0] == 'Expensive' else -1 if atm_iv['PE_IV_Bias'].iloc[0] == 'Cheap' else 0
            iv_score = (avg_ce_iv_bias + avg_pe_iv_bias) / 2
        else:
            iv_score = 0
        scores['IV'] = iv_score
    else:
        scores['IV'] = 0
    
    # Gamma Score (-1 to +1)
    if gamma_df is not None and not gamma_df.empty:
        total_net_gex = gamma_df['Net_GEX'].sum()
        if abs(total_net_gex) > 2000:
            gamma_score = 1  # High volatility expected
        elif abs(total_net_gex) < 1000:
            gamma_score = -1  # Pinning expected
        else:
            gamma_score = 0
        scores['Gamma'] = gamma_score
    else:
        scores['Gamma'] = 0
    
    # VWAP Score (-1 to +1)
    if len(df) > 0:
        df = calculate_vwap_bands(df)
        current_vwap = df['vwap'].iloc[-1]
        current_upper_1 = df['vwap_upper_1'].iloc[-1]
        current_lower_1 = df['vwap_lower_1'].iloc[-1]
        
        if current_price > current_upper_1:
            vwap_score = 1  # Bullish
        elif current_price < current_lower_1:
            vwap_score = -1  # Bearish
        else:
            vwap_score = 0  # Neutral
        scores['VWAP'] = vwap_score
    else:
        scores['VWAP'] = 0
    
    # Calculate total score
    total_score = sum(scores.values())
    max_possible = 8  # RSI(2) + OI(2) + Volume(1) + IV(1) + Gamma(1) + VWAP(1)
    
    # Determine bias
    if total_score >= 3:
        bias = "Strong Bullish"
    elif total_score >= 1:
        bias = "Bullish"
    elif total_score <= -3:
        bias = "Strong Bearish"
    elif total_score <= -1:
        bias = "Bearish"
    else:
        bias = "Neutral"
    
    # Market dynamics
    if gamma_df is not None and not gamma_df.empty:
        total_gex = abs(gamma_df['Net_GEX'].sum())
        if total_gex < 500:
            market_dynamics = "Highly Pinned"
        elif total_gex < 1500:
            market_dynamics = "Moderately Pinned"
        elif total_gex < 3000:
            market_dynamics = "Normal Volatility"
        else:
            market_dynamics = "High Volatility"
    else:
        market_dynamics = "Unknown"
    
    return bias, total_score, scores, market_dynamics

def check_advanced_signals(df, option_data, iv_df, gamma_df, volume_spread_df, current_price, proximity=5):
    """Enhanced signal detection with new features"""
    if df.empty or option_data is None or not current_price:
        return
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df)
    df = calculate_vwap_bands(df)
    df = calculate_atr(df)
    
    current_rsi = df['rsi'].iloc[-1] if not df.empty else None
    current_vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns else None
    current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
    
    # Get final bias
    bias, score, detailed_scores, dynamics = calculate_final_bias_score(
        df, option_data, iv_df, gamma_df, volume_spread_df, current_price
    )
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return
    
    row = atm_data.iloc[0]
    
    # MASTER SIGNAL - All conditions aligned (Updated with new bias columns)
    strong_bullish = (
        bias == "Strong Bullish" and
        current_rsi is not None and current_rsi < 70 and
        current_vwap is not None and current_price > current_vwap and
        row['CE_Delta_Bias'] == 'Bullish' and
        row['IV_Bias'] in ['CE Cheaper', 'Fair']
    )
    
    strong_bearish = (
        bias == "Strong Bearish" and
        current_rsi is not None and current_rsi > 30 and
        current_vwap is not None and current_price < current_vwap and
        row['PE_Delta_Bias'] == 'Bearish' and
        row['IV_Bias'] in ['PE Cheaper', 'Fair']
    )
    
    if strong_bullish or strong_bearish:
        signal_type = "STRONG CALL" if strong_bullish else "STRONG PUT"
        
        message = f"""
🔥 MASTER {signal_type} SIGNAL 🔥

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 Bias Score: {score}/8 ({bias})

📈 Technical Confluence:
RSI: {current_rsi:.2f}
VWAP: ₹{current_vwap:.2f}
ATR: {current_atr:.2f} (Stop Loss Guide)

🎲 Market Dynamics: {dynamics}
💎 All Systems Aligned!

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success(f"🔥 MASTER {signal_type} signal sent!")
    
    # VWAP BREAKOUT SIGNAL
    if current_vwap is not None and 'vwap_upper_1' in df.columns:
        upper_band = df['vwap_upper_1'].iloc[-1]
        lower_band = df['vwap_lower_1'].iloc[-1]
        
        if current_price > upper_band:
            message = f"""
🚀 VWAP BREAKOUT - BULLISH 🚀

📍 Spot: ₹{current_price:.2f}
📊 VWAP: ₹{current_vwap:.2f}
📈 Above +1σ Band: ₹{upper_band:.2f}

Strong momentum breakout above VWAP bands!

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success("🚀 VWAP Breakout signal sent!")
        
        elif current_price < lower_band:
            message = f"""
📉 VWAP BREAKDOWN - BEARISH 📉

📍 Spot: ₹{current_price:.2f}
📊 VWAP: ₹{current_vwap:.2f}
📉 Below -1σ Band: ₹{lower_band:.2f}

Strong momentum breakdown below VWAP bands!

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success("📉 VWAP Breakdown signal sent!")

def main():
    st.title("📈 Advanced Nifty Trading Analyzer")
    
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
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Enhanced Chart Analysis")
        
        data = api.get_intraday_data(interval)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        ltp_data = api.get_ltp_data()
        current_price = None
        if ltp_data and 'data' in ltp_data:
            for exchange, data_ex in ltp_data['data'].items():
                for security_id, price_data in data_ex.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        if not df.empty and len(df) > 1:
            prev_close = df['close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Calculate indicators
            df_calc = calculate_vwap_bands(df.copy())
            df_calc = calculate_atr(df_calc)
            df_calc['rsi'] = calculate_rsi(df_calc)
            
            current_rsi = df_calc['rsi'].iloc[-1] if not df_calc.empty else None
            current_vwap = df_calc['vwap'].iloc[-1] if 'vwap' in df_calc.columns else None
            current_atr = df_calc['atr'].iloc[-1] if 'atr' in df_calc.columns else None
            
            # Metrics row
            col1_m, col2_m, col3_m, col4_m, col5_m = st.columns(5)
            with col1_m:
                st.metric("Price", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            with col2_m:
                st.metric("RSI", f"{current_rsi:.2f}" if current_rsi is not None else "N/A")
            with col3_m:
                st.metric("VWAP", f"₹{current_vwap:,.2f}" if current_vwap is not None else "N/A")
            with col4_m:
                st.metric("ATR", f"{current_atr:.2f}" if current_atr is not None else "N/A")
            with col5_m:
                high_low = f"₹{df['high'].max():,.0f} / ₹{df['low'].min():,.0f}"
                st.metric("H/L", high_low)
        
        if not df.empty:
            fig = create_enhanced_chart(df, f"Nifty {interval}min")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No chart data available")
    
    with col2:
        st.header("🎯 Advanced Options Analysis")
        
        expiry_data = get_expiry_list()
        if expiry_data and 'data' in expiry_data:
            expiry_dates = expiry_data['data']
            selected_expiry = st.selectbox("Expiry", expiry_dates)
            
            if current_price:
                underlying_price, option_summary, iv_df, gamma_df, volume_spread_df = analyze_advanced_options(selected_expiry, current_price)
                
                if underlying_price and option_summary is not None:
                    st.info(f"Spot: ₹{underlying_price:.2f}")
                    
                    # Calculate final bias
                    bias, score, detailed_scores, dynamics = calculate_final_bias_score(
                        df, option_summary, iv_df, gamma_df, volume_spread_df, current_price
                    )
                    
                    # Bias Dashboard
                    st.subheader("🎪 Final Bias Dashboard")
                    bias_col1, bias_col2 = st.columns(2)
                    
                    with bias_col1:
                        bias_color = "green" if "Bullish" in bias else "red" if "Bearish" in bias else "gray"
                        st.markdown(f"**Market Bias:** <span style='color:{bias_color}'>{bias}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Score:** {score}/8")
                    
                    with bias_col2:
                        st.markdown(f"**Dynamics:** {dynamics}")
                        if current_atr:
                            stop_loss = round(current_atr * 1.5, 0)
                            st.markdown(f"**Suggested SL:** ±{stop_loss} pts")
                    
                    # Detailed scores
                    st.write("**Component Scores:**")
                    score_df = pd.DataFrame([detailed_scores])
                    st.dataframe(score_df, use_container_width=True)
                    
                    # Tabs for different analyses
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Basic OI", "📈 Volume Spread", "💫 IV Analysis", "⚡ Gamma Exposure", "🎯 Summary"])
                    
                    with tab1:
                        st.dataframe(option_summary, use_container_width=True)
                    
                    with tab2:
                        if volume_spread_df is not None and not volume_spread_df.empty:
                            st.dataframe(volume_spread_df, use_container_width=True)
                        else:
                            st.info("Volume spread data not available")
                    
                    with tab3:
                        if iv_df is not None and not iv_df.empty:
                            st.dataframe(iv_df, use_container_width=True)
                        else:
                            st.info("IV data not available")
                    
                    with tab4:
                        if gamma_df is not None and not gamma_df.empty:
                            st.dataframe(gamma_df, use_container_width=True)
                            total_gex = gamma_df['Net_GEX'].sum()
                            st.metric("Total Net GEX", f"{total_gex:,.0f}")
                        else:
                            st.info("Gamma data not available")
                    
                    with tab5:
                        # Key insights with enhanced bias analysis
                        st.write("**Key Insights:**")
                        
                        if not option_summary.empty:
                            atm_row = option_summary[option_summary['Zone'] == 'ATM'].iloc[0]
                            st.write(f"• ATM Strike: {atm_row['Strike']}")
                            st.write(f"• PCR: {atm_row['PCR']}")
                            st.write(f"• Level: {atm_row['Level']}")
                            st.write(f"• CE Delta Bias: {atm_row['CE_Delta_Bias']}")
                            st.write(f"• PE Delta Bias: {atm_row['PE_Delta_Bias']}")
                            st.write(f"• IV Bias: {atm_row['IV_Bias']} (CE: {atm_row['CE_IV']}%, PE: {atm_row['PE_IV']}%)")
                            st.write(f"• Gamma Bias: {atm_row['Gamma_Bias']} (Net GEX: {atm_row['Net_GEX']:,.0f})")
                        
                        if gamma_df is not None and not gamma_df.empty:
                            avg_gamma_bias = gamma_df['Gamma_Bias'].mode().iloc[0] if not gamma_df['Gamma_Bias'].mode().empty else "Unknown"
                            st.write(f"• Overall Volatility Expectation: {avg_gamma_bias}")
                        
                        if iv_df is not None and not iv_df.empty:
                            expensive_count = (iv_df['CE_IV_Bias'] == 'Expensive').sum() + (iv_df['PE_IV_Bias'] == 'Expensive').sum()
                            total_count = len(iv_df) * 2
                            iv_sentiment = "Expensive" if expensive_count > total_count * 0.6 else "Cheap" if expensive_count < total_count * 0.4 else "Fair"
                            st.write(f"• Options Pricing Sentiment: {iv_sentiment}")
                        
                        # Trading recommendations based on bias
                        st.write("**Trading Recommendations:**")
                        if not option_summary.empty:
                            atm_row = option_summary[option_summary['Zone'] == 'ATM'].iloc[0]
                            if atm_row['IV_Bias'] == 'CE Cheaper':
                                st.write("• Consider Call options (cheaper IV)")
                            elif atm_row['IV_Bias'] == 'PE Cheaper':
                                st.write("• Consider Put options (cheaper IV)")
                            
                            if atm_row['Gamma_Bias'] == 'Pinned':
                                st.write("• Low volatility expected - consider selling options")
                            elif atm_row['Gamma_Bias'] == 'High Vol':
                                st.write("• High volatility expected - consider buying options")
                    
                    # Signal checking
                    if enable_signals and not df.empty and is_market_hours():
                        with st.spinner("Checking for signals..."):
                            check_advanced_signals(df, option_summary, iv_df, gamma_df, volume_spread_df, current_price, proximity)
                else:
                    st.error("Options data unavailable")
            else:
                st.error("Current price not available")
        else:
            st.error("Expiry data unavailable")
    
    # Footer with status
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time}")
    
    if st.sidebar.button("🧪 Test Telegram"):
        send_telegram("🔔 Test message from Advanced Nifty Analyzer")
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()
