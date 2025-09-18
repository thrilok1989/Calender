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
import warnings
import asyncio
import websocket
import json
import threading
from collections import defaultdict, deque
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Institutional Nifty Analyzer", page_icon="🏛️", layout="wide")

# Enhanced session state management
if 'market_depth_data' not in st.session_state:
    st.session_state.market_depth_data = {}
if 'volume_flow_data' not in st.session_state:
    st.session_state.volume_flow_data = deque(maxlen=1000)
if 'option_flow_tracker' not in st.session_state:
    st.session_state.option_flow_tracker = {}
if 'portfolio_greeks' not in st.session_state:
    st.session_state.portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

# Function to check if it's market hours
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    if now.weekday() >= 5:
        return False
    
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    
    return market_start <= now <= market_end

# Auto-refresh with different intervals for different data
if is_market_hours():
    st_autorefresh(interval=15000, key="main_refresh")  # 15 seconds for main data
else:
    st.info("🏛️ Market is closed. Institutional analysis in preview mode.")

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

class InstitutionalDhanAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
        self.depth_cache = {}
        self.volume_profile = defaultdict(list)
    
    def get_market_quote_bulk(self, instruments):
        """Get market depth for multiple instruments"""
        url = "https://api.dhan.co/v2/marketfeed/quote"
        payload = instruments
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
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

def get_option_chain_detailed(expiry):
    """Enhanced option chain with full data"""
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

def send_institutional_alert(message, priority="NORMAL"):
    """Enhanced alert system for institutional signals"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    priority_emojis = {
        "LOW": "📊",
        "NORMAL": "🏛️", 
        "HIGH": "🚨",
        "CRITICAL": "🔴"
    }
    
    formatted_message = f"{priority_emojis.get(priority, '📊')} INSTITUTIONAL ALERT - {priority}\n\n{message}"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": formatted_message, "parse_mode": "HTML"}
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

# Advanced Technical Indicators
def calculate_institutional_indicators(df):
    """Calculate institutional-grade technical indicators"""
    if df.empty:
        return df
    
    # Enhanced VWAP with multiple timeframes
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_pv = (typical_price * df['volume']).cumsum()
    cumulative_volume = df['volume'].cumsum()
    
    df['vwap'] = cumulative_pv / cumulative_volume
    
    # VWAP bands (institutional levels)
    price_variance = ((typical_price - df['vwap']) ** 2 * df['volume']).cumsum() / cumulative_volume
    std_dev = np.sqrt(price_variance)
    
    df['vwap_upper_1'] = df['vwap'] + std_dev
    df['vwap_lower_1'] = df['vwap'] - std_dev
    df['vwap_upper_2'] = df['vwap'] + 2 * std_dev
    df['vwap_lower_2'] = df['vwap'] - 2 * std_dev
    df['vwap_upper_3'] = df['vwap'] + 3 * std_dev  # Extreme levels
    df['vwap_lower_3'] = df['vwap'] - 3 * std_dev
    
    # Volume-weighted RSI
    df['rsi'] = calculate_rsi(df)
    df['volume_rsi'] = calculate_volume_weighted_rsi(df)
    
    # True Range and ATR variants
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multiple ATR periods for different strategies
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    df['atr_21'] = df['true_range'].rolling(window=21).mean()
    df['atr_50'] = df['true_range'].rolling(window=50).mean()
    
    # Volume Profile Analysis
    df['volume_profile'] = calculate_volume_profile(df)
    
    # Price Action Strength
    df['price_momentum'] = calculate_price_momentum(df)
    
    return df

def calculate_rsi(data, period=14):
    """Enhanced RSI calculation"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volume_weighted_rsi(df, period=14):
    """Volume-weighted RSI for institutional flow analysis"""
    if len(df) < period + 1:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    price_change = df['close'].diff()
    volume_weighted_change = price_change * df['volume']
    
    gains = volume_weighted_change.where(volume_weighted_change > 0, 0)
    losses = -volume_weighted_change.where(volume_weighted_change < 0, 0)
    
    avg_gains = gains.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    avg_losses = losses.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    rs = avg_gains / avg_losses
    volume_rsi = 100 - (100 / (1 + rs))
    
    return volume_rsi

def calculate_volume_profile(df, bins=20):
    """Calculate volume profile for institutional accumulation/distribution"""
    if df.empty:
        return pd.Series([np.nan] * len(df), index=df.index)
    
    price_range = df['high'].max() - df['low'].min()
    bin_size = price_range / bins
    
    volume_at_price = []
    for _, row in df.iterrows():
        price_bin = int((row['close'] - df['low'].min()) / bin_size)
        price_bin = min(price_bin, bins - 1)
        volume_at_price.append(price_bin)
    
    return pd.Series(volume_at_price, index=df.index)

def calculate_price_momentum(df, period=10):
    """Calculate price momentum for trend strength"""
    if len(df) < period:
        return pd.Series([0] * len(df), index=df.index)
    
    momentum = ((df['close'] / df['close'].shift(period)) - 1) * 100
    return momentum

# Advanced Options Analysis
def analyze_institutional_options(expiry, spot_price):
    """Comprehensive institutional options analysis"""
    option_data = get_option_chain_detailed(expiry)
    if not option_data or 'data' not in option_data:
        return None, None, None, None
    
    data = option_data['data']
    underlying = data['last_price']
    oc_data = data['oc']
    
    # Calculate days to expiry
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
    current_date = datetime.now()
    days_to_expiry = max((expiry_date - current_date).days, 1)
    time_to_expiry = days_to_expiry / 365.0
    
    calls, puts = [], []
    for strike, strike_data in oc_data.items():
        if 'ce' in strike_data:
            ce_data = strike_data['ce'].copy()
            ce_data['strikePrice'] = float(strike)
            calls.append(ce_data)
        if 'pe' in strike_data:
            pe_data = strike_data['pe'].copy()
            pe_data['strikePrice'] = float(strike)
            puts.append(pe_data)
    
    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
    
    # Filter for institutional analysis (ATM ±5 strikes)
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df_filtered = df[abs(df['strikePrice'] - atm_strike) <= 250]  # ±5 strikes
    
    # Enhanced institutional metrics
    institutional_metrics = []
    total_ce_volume = total_pe_volume = 0
    total_ce_oi = total_pe_oi = 0
    gamma_exposure = 0
    vega_exposure = 0
    
    # Volatility surface data
    volatility_surface = []
    
    for _, row in df_filtered.iterrows():
        strike = row['strikePrice']
        
        # Get all available data
        ce_last_price = row.get('last_price_CE', 0)
        pe_last_price = row.get('last_price_PE', 0)
        ce_volume = row.get('volume_CE', 0)
        pe_volume = row.get('volume_PE', 0)
        ce_oi = row.get('oi_CE', 0)
        pe_oi = row.get('oi_PE', 0)
        ce_prev_oi = row.get('previous_oi_CE', 0)
        pe_prev_oi = row.get('previous_oi_PE', 0)
        
        # Greeks from API
        ce_greeks = row.get('greeks_CE', {})
        pe_greeks = row.get('greeks_PE', {})
        
        ce_delta = ce_greeks.get('delta', 0)
        pe_delta = pe_greeks.get('delta', 0)
        ce_gamma = ce_greeks.get('gamma', 0)
        pe_gamma = pe_greeks.get('gamma', 0)
        ce_theta = ce_greeks.get('theta', 0)
        pe_theta = pe_greeks.get('theta', 0)
        ce_vega = ce_greeks.get('vega', 0)
        pe_vega = pe_greeks.get('vega', 0)
        
        # IV from API
        ce_iv = row.get('implied_volatility_CE', 0) * 100 if row.get('implied_volatility_CE') else 0
        pe_iv = row.get('implied_volatility_PE', 0) * 100 if row.get('implied_volatility_PE') else 0
        
        # Advanced order flow analysis
        ce_bid_qty = row.get('top_bid_quantity_CE', 0)
        ce_ask_qty = row.get('top_ask_quantity_CE', 0)
        pe_bid_qty = row.get('top_bid_quantity_PE', 0)
        pe_ask_qty = row.get('top_ask_quantity_PE', 0)
        
        # Institutional flow classification
        ce_aggressive_buy = ce_ask_qty * 0.6 if ce_ask_qty > 0 else ce_volume * 0.3
        ce_aggressive_sell = ce_bid_qty * 0.6 if ce_bid_qty > 0 else ce_volume * 0.3
        pe_aggressive_buy = pe_ask_qty * 0.6 if pe_ask_qty > 0 else pe_volume * 0.3
        pe_aggressive_sell = pe_bid_qty * 0.6 if pe_bid_qty > 0 else pe_volume * 0.3
        
        # Volume spread delta
        ce_volume_delta = ce_aggressive_buy - ce_aggressive_sell
        pe_volume_delta = pe_aggressive_buy - pe_aggressive_sell
        
        # Gamma exposure calculation (institutional focus)
        ce_gex = ce_oi * ce_gamma * underlying * underlying * 0.01
        pe_gex = pe_oi * pe_gamma * underlying * underlying * 0.01 * (-1)
        net_gex = ce_gex + pe_gex
        
        # Vega exposure
        ce_vex = ce_oi * ce_vega
        pe_vex = pe_oi * pe_vega
        net_vex = ce_vex + pe_vex
        
        # Accumulate totals
        total_ce_volume += ce_volume
        total_pe_volume += pe_volume
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        gamma_exposure += abs(net_gex)
        vega_exposure += abs(net_vex)
        
        # Volatility surface point
        volatility_surface.append({
            'strike': strike,
            'moneyness': strike / underlying,
            'ce_iv': ce_iv,
            'pe_iv': pe_iv,
            'days_to_expiry': days_to_expiry
        })
        
        # Institutional bias analysis
        volume_bias = "Bullish" if pe_volume > ce_volume * 1.1 else "Bearish" if ce_volume > pe_volume * 1.1 else "Neutral"
        oi_bias = "Bullish" if (pe_oi - pe_prev_oi) > (ce_oi - ce_prev_oi) else "Bearish" if (ce_oi - ce_prev_oi) > (pe_oi - pe_prev_oi) else "Neutral"
        flow_bias = "Bullish" if pe_volume_delta > ce_volume_delta else "Bearish" if ce_volume_delta > pe_volume_delta else "Neutral"
        
        # Support/Resistance levels
        level = "Strong Support" if pe_oi > ce_oi * 1.5 else "Support" if pe_oi > ce_oi * 1.2 else "Strong Resistance" if ce_oi > pe_oi * 1.5 else "Resistance" if ce_oi > pe_oi * 1.2 else "Neutral"
        
        institutional_metrics.append({
            "Strike": strike,
            "Zone": 'ATM' if strike == atm_strike else 'ITM' if strike < underlying else 'OTM',
            "Level": level,
            "Volume_Bias": volume_bias,
            "OI_Bias": oi_bias,
            "Flow_Bias": flow_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else 0, 3),
            "CE_IV": round(ce_iv, 1),
            "PE_IV": round(pe_iv, 1),
            "IV_Skew": round(pe_iv - ce_iv, 1),
            "CE_Volume_Delta": round(ce_volume_delta, 0),
            "PE_Volume_Delta": round(pe_volume_delta, 0),
            "Net_GEX": round(net_gex / 1000000, 2),  # In millions
            "Net_VEX": round(net_vex / 1000, 2),     # In thousands
            "CE_Delta_Exposure": round(ce_oi * ce_delta / 1000, 2),  # In thousands
            "PE_Delta_Exposure": round(pe_oi * abs(pe_delta) / 1000, 2),
            "Total_Volume": ce_volume + pe_volume,
            "OI_Change": (ce_oi - ce_prev_oi) + (pe_oi - pe_prev_oi)
        })
    
    # Summary analytics
    summary_analytics = {
        'total_volume_ratio': total_pe_volume / total_ce_volume if total_ce_volume > 0 else 0,
        'total_oi_ratio': total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0,
        'net_gamma_exposure': gamma_exposure / 1000000,  # In millions
        'net_vega_exposure': vega_exposure / 1000,       # In thousands
        'volatility_surface': volatility_surface,
        'days_to_expiry': days_to_expiry,
        'institutional_sentiment': determine_institutional_sentiment(institutional_metrics)
    }
    
    # Risk metrics
    risk_metrics = calculate_portfolio_risk_metrics(institutional_metrics, underlying)
    
    return underlying, pd.DataFrame(institutional_metrics), summary_analytics, risk_metrics

def determine_institutional_sentiment(metrics_data):
    """Determine overall institutional sentiment"""
    bullish_signals = bearish_signals = neutral_signals = 0
    
    for metric in metrics_data:
        volume_bias = metric.get('Volume_Bias', 'Neutral')
        oi_bias = metric.get('OI_Bias', 'Neutral')
        flow_bias = metric.get('Flow_Bias', 'Neutral')
        
        biases = [volume_bias, oi_bias, flow_bias]
        bullish_signals += biases.count('Bullish')
        bearish_signals += biases.count('Bearish')
        neutral_signals += biases.count('Neutral')
    
    if bullish_signals > bearish_signals * 1.2:
        return "INSTITUTIONAL BULLISH"
    elif bearish_signals > bullish_signals * 1.2:
        return "INSTITUTIONAL BEARISH"
    else:
        return "INSTITUTIONAL NEUTRAL"

def calculate_portfolio_risk_metrics(metrics_data, spot_price):
    """Calculate portfolio-level risk metrics"""
    total_delta = sum(m.get('CE_Delta_Exposure', 0) - m.get('PE_Delta_Exposure', 0) for m in metrics_data)
    total_gamma = sum(abs(m.get('Net_GEX', 0)) for m in metrics_data)
    total_vega = sum(abs(m.get('Net_VEX', 0)) for m in metrics_data)
    
    # Risk scenarios
    price_scenarios = [0.95, 0.97, 0.99, 1.00, 1.01, 1.03, 1.05]  # ±5% range
    scenario_pnl = []
    
    for scenario in price_scenarios:
        new_price = spot_price * scenario
        price_change = new_price - spot_price
        # Simplified PnL calculation
        pnl = total_delta * price_change + 0.5 * total_gamma * (price_change ** 2)
        scenario_pnl.append(pnl)
    
    return {
        'portfolio_delta': total_delta,
        'portfolio_gamma': total_gamma,
        'portfolio_vega': total_vega,
        'max_loss_5pct': min(scenario_pnl),
        'max_gain_5pct': max(scenario_pnl),
        'scenarios': list(zip(price_scenarios, scenario_pnl))
    }

def create_institutional_dashboard_chart(df, title):
    """Create comprehensive institutional dashboard"""
    if df.empty:
        return go.Figure()
    
    # Calculate all institutional indicators
    df = calculate_institutional_indicators(df)
    
    # Create multi-panel chart
    fig = make_subplots(
        rows=5, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.2],
        column_widths=[0.7, 0.3],
        subplot_titles=[
            'Price Action with Institutional Levels', 'Volume Profile',
            'Volume Analysis', 'RSI Comparison',
            'ATR Multi-Timeframe', 'Price Momentum',
            'Order Flow Strength', 'VWAP Deviation'
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"colspan": 2}, None]
        ]
    )
    
    # Main price chart with institutional levels
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # VWAP and institutional bands
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap'], name='VWAP',
                            line=dict(color='orange', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_upper_2'], name='VWAP +2σ',
                            line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_lower_2'], name='VWAP -2σ',
                            line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_upper_3'], name='VWAP +3σ',
                            line=dict(color='darkred', width=1, dash='dot'), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_lower_3'], name='VWAP -3σ',
                            line=dict(color='darkgreen', width=1, dash='dot'), opacity=0.7), row=1, col=1)
    
    # Volume profile (simplified histogram)
    if 'volume_profile' in df.columns:
        fig.add_trace(go.Histogram(y=df['close'], nbinsy=20, name='Volume Profile',
                                  marker_color='rgba(55, 128, 191, 0.7)'), row=1, col=2)
    
    # Volume analysis
    volume_colors = ['#26a69a' if close >= open else '#ef5350'
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], name='Volume',
                        marker_color=volume_colors, opacity=0.7), row=2, col=1)
    
    # Average volume line
    avg_volume = df['volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df['datetime'], y=avg_volume, name='Avg Volume',
                            line=dict(color='purple', width=1)), row=2, col=1)
    
    # Volume strength indicator
    volume_strength = df['volume'] / avg_volume
    fig.add_trace(go.Scatter(x=df['datetime'], y=volume_strength, name='Volume Strength',
                            line=dict(color='blue', width=2)), row=2, col=2)
    fig.add_hline(y=1.5, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", row=2, col=2)
    
    # RSI comparison
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['rsi'], name='Traditional RSI',
                            line=dict(color='orange', width=2)), row=3, col=1)
    if 'volume_rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['volume_rsi'], name='Volume RSI',
                                line=dict(color='red', width=2)), row=3, col=1)
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # RSI divergence
    rsi_divergence = df['rsi'] - df.get('volume_rsi', df['rsi'])
    fig.add_trace(go.Scatter(x=df['datetime'], y=rsi_divergence, name='RSI Divergence',
                            line=dict(color='purple', width=1), fill='tonexty'), row=3, col=2)
    
    # Multi-timeframe ATR
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['atr_14'], name='ATR 14',
                            line=dict(color='blue', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['atr_21'], name='ATR 21',
                            line=dict(color='green', width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['atr_50'], name='ATR 50',
                            line=dict(color='red', width=1)), row=4, col=1)
    
    # Price momentum
    if 'price_momentum' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['price_momentum'], name='Price Momentum',
                                line=dict(color='purple', width=2), fill='tonexty'), row=4, col=2)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", row=4, col=2)
    
    # Order flow strength (bottom panel)
    current_price = df['close'].iloc[-1] if not df.empty else 0
    vwap_current = df['vwap'].iloc[-1] if 'vwap' in df.columns else 0
    vwap_deviation = ((current_price - vwap_current) / vwap_current * 100) if vwap_current > 0 else 0
    
    # Create order flow strength indicator
    order_flow_strength = calculate_order_flow_strength(df)
    fig.add_trace(go.Scatter(x=df['datetime'], y=order_flow_strength, name='Order Flow Strength',
                            line=dict(color='gold', width=3), fill='tonexty'), row=5, col=1)
    
    fig.add_hline(y=0, line_dash="solid", line_color="white", row=5, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="red", row=5, col=1)
    fig.add_hline(y=-50, line_dash="dash", line_color="green", row=5, col=1)
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update y-axis ranges
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="Order Flow", range=[-100, 100], row=5, col=1)
    
    return fig

def calculate_order_flow_strength(df, window=20):
    """Calculate institutional order flow strength"""
    if len(df) < window:
        return pd.Series([0] * len(df), index=df.index)
    
    # Volume-weighted price change
    price_change = df['close'].pct_change()
    volume_norm = df['volume'] / df['volume'].rolling(window).mean()
    
    # Order flow strength combines price momentum with volume
    flow_strength = (price_change * volume_norm * 100).rolling(window).mean()
    
    return flow_strength.fillna(0)

def create_volatility_surface_chart(volatility_data):
    """Create 3D volatility surface visualization"""
    if not volatility_data:
        return go.Figure()
    
    df_vol = pd.DataFrame(volatility_data)
    
    fig = go.Figure(data=[go.Surface(
        z=df_vol.pivot_table(values='ce_iv', index='days_to_expiry', columns='moneyness').values,
        x=df_vol['moneyness'].unique(),
        y=df_vol['days_to_expiry'].unique(),
        colorscale='RdYlBu_r',
        name='Call IV Surface'
    )])
    
    fig.update_layout(
        title='Implied Volatility Surface - Calls',
        scene=dict(
            xaxis_title='Moneyness (Strike/Spot)',
            yaxis_title='Days to Expiry',
            zaxis_title='Implied Volatility (%)'
        ),
        template='plotly_dark',
        height=500
    )
    
    return fig

def create_gamma_exposure_chart(options_data):
    """Create gamma exposure visualization"""
    if options_data is None or options_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Positive and negative GEX
    pos_gex = options_data[options_data['Net_GEX'] > 0]
    neg_gex = options_data[options_data['Net_GEX'] < 0]
    
    fig.add_trace(go.Bar(
        x=pos_gex['Strike'],
        y=pos_gex['Net_GEX'],
        name='Positive GEX',
        marker_color='green',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=neg_gex['Strike'],
        y=neg_gex['Net_GEX'],
        name='Negative GEX',
        marker_color='red',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Gamma Exposure by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Net Gamma Exposure (₹ Millions)',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_options_flow_chart(options_data):
    """Create options flow analysis chart"""
    if options_data is None or options_data.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Volume Delta Flow', 'OI vs Volume', 'IV Skew', 'PCR Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Volume Delta Flow
    fig.add_trace(go.Bar(
        x=options_data['Strike'],
        y=options_data['CE_Volume_Delta'],
        name='CE Volume Delta',
        marker_color='blue',
        opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=options_data['Strike'],
        y=options_data['PE_Volume_Delta'],
        name='PE Volume Delta',
        marker_color='red',
        opacity=0.7
    ), row=1, col=1)
    
    # OI vs Volume scatter
    fig.add_trace(go.Scatter(
        x=options_data['Total_Volume'],
        y=options_data['OI_Change'],
        mode='markers+text',
        text=options_data['Strike'],
        name='OI Change vs Volume',
        marker=dict(size=8, color=options_data['PCR'], colorscale='RdYlGn', showscale=True)
    ), row=1, col=2)
    
    # IV Skew
    fig.add_trace(go.Scatter(
        x=options_data['Strike'],
        y=options_data['CE_IV'],
        name='Call IV',
        line=dict(color='blue', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=options_data['Strike'],
        y=options_data['PE_IV'],
        name='Put IV',
        line=dict(color='red', width=2)
    ), row=2, col=1)
    
    # PCR Analysis
    fig.add_trace(go.Bar(
        x=options_data['Strike'],
        y=options_data['PCR'],
        name='PCR',
        marker_color='purple',
        opacity=0.7
    ), row=2, col=2)
    
    fig.add_hline(y=1, line_dash="dash", line_color="white", row=2, col=2)
    
    fig.update_layout(
        title='Institutional Options Flow Analysis',
        template='plotly_dark',
        height=800,
        showlegend=True
    )
    
    return fig

def detect_institutional_signals(df, options_data, summary_analytics, current_price):
    """Detect institutional-level trading signals"""
    signals = []
    
    if df.empty or options_data is None:
        return signals
    
    # Calculate current indicators
    df = calculate_institutional_indicators(df)
    current_vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns else current_price
    current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
    current_volume_rsi = df.get('volume_rsi', pd.Series([50])).iloc[-1]
    
    # 1. VWAP Institutional Levels Signal
    vwap_deviation = ((current_price - current_vwap) / current_vwap) * 100
    
    if abs(vwap_deviation) > 0.5:  # 0.5% deviation from VWAP
        if 'vwap_upper_2' in df.columns and current_price > df['vwap_upper_2'].iloc[-1]:
            signals.append({
                'type': 'INSTITUTIONAL_BREAKOUT',
                'direction': 'BULLISH',
                'strength': 'HIGH',
                'message': f'Price above VWAP +2σ band. Institutional breakout likely. Deviation: {vwap_deviation:.2f}%',
                'level': df['vwap_upper_2'].iloc[-1]
            })
        elif 'vwap_lower_2' in df.columns and current_price < df['vwap_lower_2'].iloc[-1]:
            signals.append({
                'type': 'INSTITUTIONAL_BREAKDOWN',
                'direction': 'BEARISH',
                'strength': 'HIGH',
                'message': f'Price below VWAP -2σ band. Institutional selling pressure. Deviation: {vwap_deviation:.2f}%',
                'level': df['vwap_lower_2'].iloc[-1]
            })
    
    # 2. Volume RSI Divergence Signal
    rsi_divergence = current_rsi - current_volume_rsi
    if abs(rsi_divergence) > 15:  # Significant divergence
        direction = "BEARISH" if rsi_divergence > 0 else "BULLISH"
        signals.append({
            'type': 'VOLUME_RSI_DIVERGENCE',
            'direction': direction,
            'strength': 'MEDIUM',
            'message': f'RSI-Volume RSI divergence: {rsi_divergence:.1f}. Institutional flow contrary to price.',
            'rsi': current_rsi,
            'volume_rsi': current_volume_rsi
        })
    
    # 3. Gamma Exposure Signals
    net_gex = summary_analytics.get('net_gamma_exposure', 0)
    if net_gex > 100:  # High gamma environment
        signals.append({
            'type': 'HIGH_GAMMA_ENVIRONMENT',
            'direction': 'NEUTRAL',
            'strength': 'HIGH',
            'message': f'High Gamma Exposure: ₹{net_gex:.0f}M. Expect low volatility/pinning near strikes.',
            'gamma_exposure': net_gex
        })
    elif net_gex < 20:  # Low gamma environment
        signals.append({
            'type': 'LOW_GAMMA_ENVIRONMENT',
            'direction': 'NEUTRAL',
            'strength': 'HIGH',
            'message': f'Low Gamma Exposure: ₹{net_gex:.0f}M. Higher volatility expected.',
            'gamma_exposure': net_gex
        })
    
    # 4. Institutional Sentiment Signal
    institutional_sentiment = summary_analytics.get('institutional_sentiment', 'NEUTRAL')
    if 'BULLISH' in institutional_sentiment or 'BEARISH' in institutional_sentiment:
        direction = 'BULLISH' if 'BULLISH' in institutional_sentiment else 'BEARISH'
        signals.append({
            'type': 'INSTITUTIONAL_SENTIMENT',
            'direction': direction,
            'strength': 'HIGH',
            'message': f'Institutional sentiment: {institutional_sentiment}',
            'sentiment': institutional_sentiment
        })
    
    # 5. Volume Profile Signals
    atm_data = options_data[options_data['Zone'] == 'ATM']
    if not atm_data.empty:
        atm_row = atm_data.iloc[0]
        total_volume = atm_row.get('Total_Volume', 0)
        avg_volume = options_data['Total_Volume'].mean()
        
        if total_volume > avg_volume * 2:  # High volume at ATM
            signals.append({
                'type': 'ATM_HIGH_VOLUME',
                'direction': 'NEUTRAL',
                'strength': 'MEDIUM',
                'message': f'Unusually high volume at ATM strike {atm_row["Strike"]}. Volume: {total_volume:,}',
                'volume': total_volume
            })
    
    # 6. Volatility Surface Signals
    volatility_surface = summary_analytics.get('volatility_surface', [])
    if volatility_surface:
        avg_iv = np.mean([v['ce_iv'] + v['pe_iv'] for v in volatility_surface]) / 2
        if avg_iv > 25:
            signals.append({
                'type': 'HIGH_VOLATILITY_REGIME',
                'direction': 'NEUTRAL',
                'strength': 'MEDIUM',
                'message': f'High IV environment: {avg_iv:.1f}%. Consider volatility selling strategies.',
                'avg_iv': avg_iv
            })
        elif avg_iv < 12:
            signals.append({
                'type': 'LOW_VOLATILITY_REGIME',
                'direction': 'NEUTRAL',
                'strength': 'MEDIUM',
                'message': f'Low IV environment: {avg_iv:.1f}%. Consider volatility buying strategies.',
                'avg_iv': avg_iv
            })
    
    return signals

def send_institutional_signals(signals, current_price):
    """Send institutional signals via Telegram"""
    if not signals:
        return
    
    for signal in signals:
        priority = "HIGH" if signal['strength'] == 'HIGH' else "NORMAL"
        
        message = f"""
<b>{signal['type']}</b> - {signal['direction']}
Strength: {signal['strength']}

{signal['message']}

Current Price: ₹{current_price:.2f}
Time: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        
        send_institutional_alert(message, priority)

def calculate_portfolio_metrics(options_data):
    """Calculate real-time portfolio metrics"""
    if options_data is None or options_data.empty:
        return {}
    
    # Aggregate portfolio Greeks
    total_delta_exposure = options_data['CE_Delta_Exposure'].sum() - options_data['PE_Delta_Exposure'].sum()
    total_gamma_exposure = options_data['Net_GEX'].abs().sum()
    total_vega_exposure = options_data['Net_VEX'].abs().sum()
    
    # Risk metrics
    max_gamma_strike = options_data.loc[options_data['Net_GEX'].abs().idxmax(), 'Strike'] if not options_data.empty else 0
    dominant_flow = "PUT" if options_data['PE_Volume_Delta'].sum() > options_data['CE_Volume_Delta'].sum() else "CALL"
    
    return {
        'portfolio_delta': total_delta_exposure,
        'portfolio_gamma': total_gamma_exposure,
        'portfolio_vega': total_vega_exposure,
        'max_gamma_strike': max_gamma_strike,
        'dominant_flow': dominant_flow,
        'total_volume': options_data['Total_Volume'].sum(),
        'avg_pcr': options_data['PCR'].mean()
    }

def main():
    st.title("🏛️ Institutional-Level Nifty Trading Analyzer")
    st.markdown("*Advanced order flow, options analytics, and risk management for institutional trading*")
    
    # Show market status
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"🏛️ Market closed. Current time: {current_time.strftime('%H:%M:%S IST')}")
        st.info("Institutional analysis available in preview mode with latest data")
    
    # Enhanced sidebar
    st.sidebar.header("📊 Institutional Settings")
    
    # Analysis parameters
    timeframe = st.sidebar.selectbox("Analysis Timeframe", ["1", "3", "5", "15"], index=2)
    depth_levels = st.sidebar.selectbox("Market Depth Analysis", ["Standard (5-level)", "Advanced (20-level)"], index=1)
    risk_scenario = st.sidebar.slider("Risk Scenario Range (%)", 1, 10, 5)
    enable_alerts = st.sidebar.checkbox("Institutional Alerts", value=True)
    
    # Advanced filters
    st.sidebar.subheader("🎯 Focus Areas")
    analyze_flow = st.sidebar.checkbox("Order Flow Analysis", value=True)
    analyze_volatility = st.sidebar.checkbox("Volatility Surface", value=True)
    analyze_gamma = st.sidebar.checkbox("Gamma Exposure", value=True)
    portfolio_mode = st.sidebar.checkbox("Portfolio Mode", value=False)
    
    # Initialize API
    api = InstitutionalDhanAPI()
    
    # Create main layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.header("📈 Institutional Price Action")
        
        # Get and process data
        data = api.get_intraday_data(timeframe, days_back=2)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        # Get current price
        current_price = df['close'].iloc[-1] if not df.empty else 0
        
        if not df.empty and len(df) > 1:
            prev_close = df['close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Calculate institutional indicators
            df_enhanced = calculate_institutional_indicators(df)
            current_vwap = df_enhanced['vwap'].iloc[-1] if 'vwap' in df_enhanced.columns else current_price
            vwap_deviation = ((current_price - current_vwap) / current_vwap) * 100
            
            # Display key metrics
            col1_m, col2_m, col3_m, col4_m = st.columns(4)
            with col1_m:
                st.metric("Nifty", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            with col2_m:
                st.metric("VWAP", f"₹{current_vwap:.2f}", f"{vwap_deviation:+.3f}%")
            with col3_m:
                current_atr = df_enhanced.get('atr_14', pd.Series([0])).iloc[-1]
                atr_pct = (current_atr / current_price) * 100 if current_price > 0 else 0
                st.metric("ATR %", f"{atr_pct:.2f}%", f"₹{current_atr:.2f}")
            with col4_m:
                volume_strength = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 1
                st.metric("Vol Strength", f"{volume_strength:.2f}x")
        
        # Create institutional dashboard
        if not df.empty:
            fig = create_institutional_dashboard_chart(df, f"Institutional Analysis - {timeframe}min")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("📊 No chart data available")
    
    with col2:
        st.header("📋 Options Analytics")
        
        # Get expiry data
        expiry_data = get_expiry_list()
        if expiry_data and 'data' in expiry_data:
            expiry_dates = expiry_data['data']
            selected_expiry = st.selectbox("Expiry Date", expiry_dates)
            
            # Analyze options with institutional metrics
            underlying_price, options_summary, summary_analytics, risk_metrics = analyze_institutional_options(selected_expiry, current_price)
            
            if underlying_price and options_summary is not None:
                st.info(f"**Underlying**: ₹{underlying_price:.2f}")
                
                # Key institutional metrics
                institutional_sentiment = summary_analytics.get('institutional_sentiment', 'NEUTRAL')
                days_to_expiry = summary_analytics.get('days_to_expiry', 0)
                net_gex = summary_analytics.get('net_gamma_exposure', 0)
                
                st.markdown(f"""
                **Institutional Sentiment**: {institutional_sentiment}  
                **Days to Expiry**: {days_to_expiry}  
                **Net Gamma Exposure**: ₹{net_gex:.1f}M  
                **Volume Ratio (P/C)**: {summary_analytics.get('total_volume_ratio', 0):.2f}
                """)
                
                # Enhanced options data display
                display_columns = ['Strike', 'Zone', 'Level', 'PCR', 'CE_IV', 'PE_IV', 'IV_Skew', 'Net_GEX', 'Total_Volume']
                st.dataframe(
                    options_summary[display_columns].style.format({
                        'PCR': '{:.3f}',
                        'CE_IV': '{:.1f}%',
                        'PE_IV': '{:.1f}%',
                        'IV_Skew': '{:.1f}%',
                        'Net_GEX': '₹{:.1f}M',
                        'Total_Volume': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Generate and send institutional signals
                if enable_alerts and is_market_hours():
                    signals = detect_institutional_signals(df, options_summary, summary_analytics, current_price)
                    if signals:
                        send_institutional_signals(signals, current_price)
                        for signal in signals[-3:]:  # Show last 3 signals
                            signal_color = "🔴" if signal['strength'] == 'HIGH' else "🟡"
                            st.success(f"{signal_color} {signal['type']}: {signal['message']}")
            
            else:
                st.error("📊 Options data unavailable")
        else:
            st.error("📅 Expiry data unavailable")
    
    with col3:
        st.header("⚖️ Risk Management")
        
        if 'options_summary' in locals() and options_summary is not None:
            # Portfolio metrics
            portfolio_metrics = calculate_portfolio_metrics(options_summary)
            
            st.subheader("Portfolio Greeks")
            st.metric("Net Delta", f"₹{portfolio_metrics.get('portfolio_delta', 0):,.0f}")
            st.metric("Total Gamma", f"₹{portfolio_metrics.get('portfolio_gamma', 0):.1f}M")
            st.metric("Total Vega", f"₹{portfolio_metrics.get('portfolio_vega', 0):,.0f}")
            
            # Risk scenarios
            if 'risk_metrics' in locals() and risk_metrics:
                st.subheader("Risk Scenarios")
                max_loss = risk_metrics.get('max_loss_5pct', 0)
                max_gain = risk_metrics.get('max_gain_5pct', 0)
                
                st.metric("Max Loss (5%)", f"₹{max_loss:,.0f}", delta_color="inverse")
                st.metric("Max Gain (5%)", f"₹{max_gain:,.0f}")
            
            # Key levels
            st.subheader("Key Levels")
            if not df.empty and 'vwap_upper_2' in df.columns:
                vwap_upper = df['vwap_upper_2'].iloc[-1]
                vwap_lower = df['vwap_lower_2'].iloc[-1]
                st.write(f"**VWAP +2σ**: ₹{vwap_upper:.2f}")
                st.write(f"**VWAP -2σ**: ₹{vwap_lower:.2f}")
            
            max_gamma_strike = portfolio_metrics.get('max_gamma_strike', 0)
            if max_gamma_strike > 0:
                st.write(f"**Max Gamma Strike**: {max_gamma_strike}")
        
        # Institutional flow summary
        st.subheader("Flow Summary")
        if 'portfolio_metrics' in locals():
            dominant_flow = portfolio_metrics.get('dominant_flow', 'NEUTRAL')
            avg_pcr = portfolio_metrics.get('avg_pcr', 0)
            
            flow_color = "🔴" if dominant_flow == "PUT" else "🟢" if dominant_flow == "CALL" else "🟡"
            st.write(f"{flow_color} **Dominant Flow**: {dominant_flow}")
            st.write(f"📊 **Average PCR**: {avg_pcr:.3f}")
    
    # Additional analysis sections
    if analyze_volatility and 'summary_analytics' in locals():
        st.header("📊 Volatility Surface Analysis")
        volatility_surface = summary_analytics.get('volatility_surface', [])
        if volatility_surface:
            vol_fig = create_volatility_surface_chart(volatility_surface)
            st.plotly_chart(vol_fig, use_container_width=True)
    
    if analyze_gamma and 'options_summary' in locals() and options_summary is not None:
        st.header("🎯 Gamma Exposure Analysis")
        gamma_fig = create_gamma_exposure_chart(options_summary)
        st.plotly_chart(gamma_fig, use_container_width=True)
    
    if analyze_flow and 'options_summary' in locals() and options_summary is not None:
        st.header("🌊 Options Flow Analysis")
        flow_fig = create_options_flow_chart(options_summary)
        st.plotly_chart(flow_fig, use_container_width=True)
    
    # Footer with update time
    current_time_str = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"🔄 Updated: {current_time_str}")
    
    if st.sidebar.button("📨 Test Institutional Alert"):
        send_institutional_alert("🏛️ Test message from Institutional Nifty Analyzer", "NORMAL")
        st.sidebar.success("Alert sent!")

if __name__ == "__main__":
    main()