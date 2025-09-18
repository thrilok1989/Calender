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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Real Institutional Nifty Analyzer", page_icon="🏛️", layout="wide")

# Function to check if it's market hours
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    if now.weekday() >= 5:
        return False
    
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    
    return market_start <= now <= market_end

# Auto-refresh during market hours
if is_market_hours():
    st_autorefresh(interval=20000, key="main_refresh")  # 20 seconds
else:
    st.info("Market is closed. Analysis in preview mode.")

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

class RealInstitutionalAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
    
    def get_market_depth(self, instruments):
        """Get real market depth data"""
        url = "https://api.dhan.co/v2/marketfeed/quote"
        try:
            response = requests.post(url, headers=self.headers, json=instruments)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Market depth error: {str(e)}")
            return None
    
    def get_ltp_data(self):
        """Get current price"""
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = {NIFTY_SEG: [NIFTY_SCRIP]}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"LTP error: {str(e)}")
            return None
    
    def get_ohlc_data(self, instruments):
        """Get OHLC data"""
        url = "https://api.dhan.co/v2/marketfeed/ohlc"
        try:
            response = requests.post(url, headers=self.headers, json=instruments)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"OHLC error: {str(e)}")
            return None
    
    def get_intraday_data(self, interval="5", days_back=2):
        """Get historical data"""
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
        except Exception as e:
            st.error(f"Historical data error: {str(e)}")
            return None

def get_option_chain(expiry):
    """Get real option chain data"""
    url = "https://api.dhan.co/v2/optionchain"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Option chain error: {str(e)}")
        return None

def get_expiry_list():
    """Get expiry dates"""
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Expiry list error: {str(e)}")
        return None

def send_alert(message, priority="NORMAL"):
    """Send Telegram alert"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    priority_emojis = {"LOW": "📊", "NORMAL": "🏛️", "HIGH": "🚨", "CRITICAL": "🔴"}
    formatted_message = f"{priority_emojis.get(priority, '📊')} INSTITUTIONAL ALERT - {priority}\n\n{message}"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": formatted_message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def process_candle_data(data):
    """Process historical data"""
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

def process_real_market_depth(depth_data):
    """Process actual market depth data from API"""
    if not depth_data or 'data' not in depth_data:
        return None
    
    processed = {}
    
    for exchange, instruments in depth_data['data'].items():
        for scrip_id, depth_info in instruments.items():
            analysis = {
                'last_price': depth_info.get('last_price', 0),
                'volume': depth_info.get('volume', 0),
                'buy_quantity': depth_info.get('buy_quantity', 0),
                'sell_quantity': depth_info.get('sell_quantity', 0),
                'average_price': depth_info.get('average_price', 0),
                'net_change': depth_info.get('net_change', 0),
                'last_quantity': depth_info.get('last_quantity', 0),
                'ohlc': depth_info.get('ohlc', {}),
                'depth_levels': []
            }
            
            # Process actual depth levels from API
            if 'depth' in depth_info:
                buy_depth = depth_info['depth'].get('buy', [])
                sell_depth = depth_info['depth'].get('sell', [])
                
                # Calculate real depth metrics
                total_bid_qty = sum(level.get('quantity', 0) for level in buy_depth[:5] if level.get('price', 0) > 0)
                total_ask_qty = sum(level.get('quantity', 0) for level in sell_depth[:5] if level.get('price', 0) > 0)
                
                analysis['total_bid_qty'] = total_bid_qty
                analysis['total_ask_qty'] = total_ask_qty
                analysis['depth_imbalance'] = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty) * 100 if (total_bid_qty + total_ask_qty) > 0 else 0
                
                # Best bid/ask
                if buy_depth and buy_depth[0].get('price', 0) > 0:
                    analysis['best_bid'] = buy_depth[0]['price']
                    analysis['best_bid_qty'] = buy_depth[0]['quantity']
                
                if sell_depth and sell_depth[0].get('price', 0) > 0:
                    analysis['best_ask'] = sell_depth[0]['price']
                    analysis['best_ask_qty'] = sell_depth[0]['quantity']
                
                # Bid-ask spread
                if 'best_bid' in analysis and 'best_ask' in analysis:
                    analysis['bid_ask_spread'] = analysis['best_ask'] - analysis['best_bid']
                    analysis['bid_ask_spread_pct'] = (analysis['bid_ask_spread'] / analysis['best_ask']) * 100
                
                # Store actual depth levels
                for i in range(min(5, len(buy_depth), len(sell_depth))):
                    if buy_depth[i].get('price', 0) > 0 and sell_depth[i].get('price', 0) > 0:
                        analysis['depth_levels'].append({
                            'level': i + 1,
                            'bid_price': buy_depth[i].get('price', 0),
                            'bid_qty': buy_depth[i].get('quantity', 0),
                            'bid_orders': buy_depth[i].get('orders', 0),
                            'ask_price': sell_depth[i].get('price', 0),
                            'ask_qty': sell_depth[i].get('quantity', 0),
                            'ask_orders': sell_depth[i].get('orders', 0)
                        })
            
            processed[f"{exchange}_{scrip_id}"] = analysis
    
    return processed

def calculate_real_technical_indicators(df):
    """Calculate technical indicators using actual data"""
    if df.empty or len(df) < 20:
        return df
    
    # Real VWAP calculation
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_pv = (typical_price * df['volume']).cumsum()
    cumulative_volume = df['volume'].cumsum()
    
    df['vwap'] = cumulative_pv / cumulative_volume
    
    # VWAP standard deviation bands
    price_variance = ((typical_price - df['vwap']) ** 2 * df['volume']).cumsum() / cumulative_volume
    std_dev = np.sqrt(price_variance)
    
    df['vwap_upper_1'] = df['vwap'] + std_dev
    df['vwap_lower_1'] = df['vwap'] - std_dev
    df['vwap_upper_2'] = df['vwap'] + 2 * std_dev
    df['vwap_lower_2'] = df['vwap'] - 2 * std_dev
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR calculation
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    
    return df

def analyze_real_options(expiry, current_price):
    """Analyze options using only real API data"""
    option_data = get_option_chain(expiry)
    if not option_data or 'data' not in option_data:
        return None, None, None
    
    data = option_data['data']
    underlying = data['last_price']
    oc_data = data['oc']
    
    # Calculate real days to expiry
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
    current_date = datetime.now()
    days_to_expiry = max((expiry_date - current_date).days, 1)
    
    # Process real option data - handle strikes more carefully
    options_analysis = []
    total_ce_volume = total_pe_volume = 0
    total_ce_oi = total_pe_oi = 0
    total_ce_oi_change = total_pe_oi_change = 0
    
    # Get available strikes from the actual API response
    available_strikes = []
    for strike_key, strike_data in oc_data.items():
        try:
            strike_value = float(strike_key)
            available_strikes.append(strike_value)
        except (ValueError, TypeError):
            continue
    
    if not available_strikes:
        return None, None, None
    
    # Find ATM strike from available strikes
    atm_strike = min(available_strikes, key=lambda x: abs(x - underlying))
    
    # Filter ATM ±3 strikes for focused analysis
    relevant_strikes = [s for s in available_strikes if abs(s - atm_strike) <= 150]
    
    for strike in relevant_strikes:
        # Use the exact string key from the API response
        strike_key = None
        for key in oc_data.keys():
            try:
                if abs(float(key) - strike) < 0.1:  # Handle floating point precision
                    strike_key = key
                    break
            except (ValueError, TypeError):
                continue
        
        if not strike_key:
            continue
            
        strike_data = oc_data[strike_key]
        
        # Check if both CE and PE data exist
        if 'ce' not in strike_data or 'pe' not in strike_data:
            continue
            
        ce_data = strike_data['ce']
        pe_data = strike_data['pe']
        
        # Extract real data from API with safe defaults
        ce_ltp = ce_data.get('last_price', 0)
        pe_ltp = pe_data.get('last_price', 0)
        ce_volume = ce_data.get('volume', 0)
        pe_volume = pe_data.get('volume', 0)
        ce_oi = ce_data.get('oi', 0)
        pe_oi = pe_data.get('oi', 0)
        ce_prev_oi = ce_data.get('previous_oi', 0)
        pe_prev_oi = pe_data.get('previous_oi', 0)
        
        # Real OI changes
        ce_oi_change = ce_oi - ce_prev_oi
        pe_oi_change = pe_oi - pe_prev_oi
        
        # Real Greeks from API with safe handling
        ce_greeks = ce_data.get('greeks', {}) or {}
        pe_greeks = pe_data.get('greeks', {}) or {}
        
        ce_delta = ce_greeks.get('delta', 0) if ce_greeks else 0
        pe_delta = pe_greeks.get('delta', 0) if pe_greeks else 0
        ce_gamma = ce_greeks.get('gamma', 0) if ce_greeks else 0
        pe_gamma = pe_greeks.get('gamma', 0) if pe_greeks else 0
        ce_vega = ce_greeks.get('vega', 0) if ce_greeks else 0
        pe_vega = pe_greeks.get('vega', 0) if pe_greeks else 0
        
        # Real IV from API with safe handling
        ce_iv = ce_data.get('implied_volatility', 0)
        pe_iv = pe_data.get('implied_volatility', 0)
        ce_iv = (ce_iv * 100) if ce_iv else 0
        pe_iv = (pe_iv * 100) if pe_iv else 0
        
        # Real bid-ask data
        ce_bid_qty = ce_data.get('top_bid_quantity', 0)
        ce_ask_qty = ce_data.get('top_ask_quantity', 0)
        pe_bid_qty = pe_data.get('top_bid_quantity', 0)
        pe_ask_qty = pe_data.get('top_ask_quantity', 0)
        
        # Calculate real metrics
        pcr_oi = pe_oi / ce_oi if ce_oi > 0 else 0
        pcr_volume = pe_volume / ce_volume if ce_volume > 0 else 0
        
        # Real gamma exposure calculation
        if underlying > 0:
            ce_gex = ce_oi * ce_gamma * underlying * underlying * 0.01
            pe_gex = pe_oi * pe_gamma * underlying * underlying * 0.01 * (-1)  # PE GEX is negative
            net_gex = ce_gex + pe_gex
        else:
            net_gex = 0
        
        # Accumulate totals
        total_ce_volume += ce_volume
        total_pe_volume += pe_volume
        total_ce_oi += ce_oi
        total_pe_oi += pe_oi
        total_ce_oi_change += ce_oi_change
        total_pe_oi_change += pe_oi_change
        
        # Determine levels based on real OI
        if pe_oi > ce_oi * 1.5:
            level = "Strong Support"
        elif pe_oi > ce_oi * 1.2:
            level = "Support"
        elif ce_oi > pe_oi * 1.5:
            level = "Strong Resistance"
        elif ce_oi > pe_oi * 1.2:
            level = "Resistance"
        else:
            level = "Neutral"
        
        options_analysis.append({
            "Strike": strike,
            "Zone": 'ATM' if strike == atm_strike else 'ITM' if strike < underlying else 'OTM',
            "Level": level,
            "CE_LTP": ce_ltp,
            "PE_LTP": pe_ltp,
            "CE_Volume": ce_volume,
            "PE_Volume": pe_volume,
            "CE_OI": ce_oi,
            "PE_OI": pe_oi,
            "CE_OI_Change": ce_oi_change,
            "PE_OI_Change": pe_oi_change,
            "PCR_OI": round(pcr_oi, 3),
            "PCR_Volume": round(pcr_volume, 3),
            "CE_IV": round(ce_iv, 1),
            "PE_IV": round(pe_iv, 1),
            "IV_Skew": round(pe_iv - ce_iv, 1),
            "CE_Delta": round(ce_delta, 3),
            "PE_Delta": round(pe_delta, 3),
            "CE_Gamma": round(ce_gamma, 4),
            "PE_Gamma": round(pe_gamma, 4),
            "Net_GEX": round(net_gex / 1000000, 2) if net_gex != 0 else 0,  # In millions
            "CE_Bid_Qty": ce_bid_qty,
            "CE_Ask_Qty": ce_ask_qty,
            "PE_Bid_Qty": pe_bid_qty,
            "PE_Ask_Qty": pe_ask_qty
        })
    
    if not options_analysis:
        return None, None, None
    
    # Real summary metrics
    summary = {
        'total_ce_volume': total_ce_volume,
        'total_pe_volume': total_pe_volume,
        'total_ce_oi': total_ce_oi,
        'total_pe_oi': total_pe_oi,
        'pcr_volume': total_pe_volume / total_ce_volume if total_ce_volume > 0 else 0,
        'pcr_oi': total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0,
        'net_oi_change': total_ce_oi_change + total_pe_oi_change,
        'days_to_expiry': days_to_expiry
    }
    
    return underlying, pd.DataFrame(options_analysis), summary

def create_real_market_depth_chart(depth_data):
    """Create chart using real market depth data"""
    if not depth_data:
        return go.Figure()
    
    first_key = list(depth_data.keys())[0]
    depth_info = depth_data[first_key]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Real Market Depth - Bid/Ask Levels', 'Depth Metrics'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Plot real depth levels
    if depth_info['depth_levels']:
        levels = depth_info['depth_levels']
        
        bid_prices = [level['bid_price'] for level in levels]
        bid_qtys = [level['bid_qty'] for level in levels]
        ask_prices = [level['ask_price'] for level in levels]
        ask_qtys = [level['ask_qty'] for level in levels]
        
        # Bid side (green)
        fig.add_trace(go.Bar(
            x=bid_qtys,
            y=bid_prices,
            orientation='h',
            name='Bids',
            marker_color='green',
            opacity=0.7,
            text=[f"₹{p:.1f} ({q:,})" for p, q in zip(bid_prices, bid_qtys)],
            textposition='inside'
        ), row=1, col=1)
        
        # Ask side (red, negative for left side)
        fig.add_trace(go.Bar(
            x=[-qty for qty in ask_qtys],
            y=ask_prices,
            orientation='h',
            name='Asks',
            marker_color='red',
            opacity=0.7,
            text=[f"₹{p:.1f} ({q:,})" for p, q in zip(ask_prices, ask_qtys)],
            textposition='inside'
        ), row=1, col=1)
    
    # Real metrics table
    metrics = {
        'Last Price': f"₹{depth_info.get('last_price', 0):.2f}",
        'Volume': f"{depth_info.get('volume', 0):,}",
        'Depth Imbalance': f"{depth_info.get('depth_imbalance', 0):.1f}%",
        'Bid-Ask Spread': f"₹{depth_info.get('bid_ask_spread', 0):.2f}",
        'Best Bid': f"₹{depth_info.get('best_bid', 0):.2f} ({depth_info.get('best_bid_qty', 0):,})",
        'Best Ask': f"₹{depth_info.get('best_ask', 0):.2f} ({depth_info.get('best_ask_qty', 0):,})"
    }
    
    # Add metrics as annotations
    y_pos = 0.8
    for i, (metric, value) in enumerate(metrics.items()):
        fig.add_annotation(
            x=0.1, y=y_pos - i * 0.12,
            text=f"<b>{metric}:</b> {value}",
            xref="x domain", yref="y domain",
            showarrow=False,
            font=dict(size=11),
            row=2, col=1,
            xanchor='left'
        )
    
    fig.update_layout(
        title='Real-time Market Depth Analysis',
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    return fig

def create_real_price_chart(df, title):
    """Create price chart with real indicators"""
    if df.empty:
        return go.Figure()
    
    df = calculate_real_technical_indicators(df)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price with VWAP Bands', 'Volume', 'RSI')
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # VWAP and bands
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap'], name='VWAP',
                            line=dict(color='orange', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_upper_2'], name='VWAP +2σ',
                            line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_lower_2'], name='VWAP -2σ',
                            line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    
    # Volume
    volume_colors = ['#26a69a' if close >= open else '#ef5350'
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], name='Volume',
                        marker_color=volume_colors, opacity=0.7), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['rsi'], name='RSI',
                            line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(title=title, template='plotly_dark', height=800, showlegend=True)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    return fig

def create_real_options_chart(options_data):
    """Create options analysis using real data"""
    if options_data is None or options_data.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Open Interest Analysis', 'Volume Analysis', 'Greeks Analysis', 'IV Analysis')
    )
    
    # OI Analysis
    fig.add_trace(go.Bar(x=options_data['Strike'], y=options_data['CE_OI'], name='CE OI',
                        marker_color='blue', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Bar(x=options_data['Strike'], y=options_data['PE_OI'], name='PE OI',
                        marker_color='red', opacity=0.7), row=1, col=1)
    
    # Volume Analysis
    fig.add_trace(go.Bar(x=options_data['Strike'], y=options_data['CE_Volume'], name='CE Vol',
                        marker_color='lightblue', opacity=0.7), row=1, col=2)
    fig.add_trace(go.Bar(x=options_data['Strike'], y=options_data['PE_Volume'], name='PE Vol',
                        marker_color='lightcoral', opacity=0.7), row=1, col=2)
    
    # Gamma Analysis
    fig.add_trace(go.Scatter(x=options_data['Strike'], y=options_data['CE_Gamma'], name='CE Gamma',
                            line=dict(color='blue', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=options_data['Strike'], y=options_data['PE_Gamma'], name='PE Gamma',
                            line=dict(color='red', width=2)), row=2, col=1)
    
    # IV Analysis
    fig.add_trace(go.Scatter(x=options_data['Strike'], y=options_data['CE_IV'], name='CE IV',
                            line=dict(color='darkblue', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=options_data['Strike'], y=options_data['PE_IV'], name='PE IV',
                            line=dict(color='darkred', width=2)), row=2, col=2)
    
    fig.update_layout(title='Real Options Analysis', template='plotly_dark', height=600, showlegend=True)
    
    return fig

def detect_real_signals(df, options_data, summary_metrics, current_price, depth_data=None):
    """Detect signals using only real data"""
    signals = []
    
    if df.empty or options_data is None:
        return signals
    
    df = calculate_real_technical_indicators(df)
    
    # Real VWAP signal
    if 'vwap' in df.columns and len(df) > 0:
        current_vwap = df['vwap'].iloc[-1]
        vwap_deviation = ((current_price - current_vwap) / current_vwap) * 100
        
        if abs(vwap_deviation) > 0.75:  # Significant deviation
            if 'vwap_upper_2' in df.columns and current_price > df['vwap_upper_2'].iloc[-1]:
                signals.append({
                    'type': 'VWAP_BREAKOUT',
                    'direction': 'BULLISH',
                    'strength': 'HIGH',
                    'message': f'Price broke above VWAP +2σ. Current deviation: {vwap_deviation:.2f}%',
                    'current_price': current_price,
                    'vwap_level': df['vwap_upper_2'].iloc[-1]
                })
            elif 'vwap_lower_2' in df.columns and current_price < df['vwap_lower_2'].iloc[-1]:
                signals.append({
                    'type': 'VWAP_BREAKDOWN',
                    'direction': 'BEARISH',
                    'strength': 'HIGH',
                    'message': f'Price broke below VWAP -2σ. Current deviation: {vwap_deviation:.2f}%',
                    'current_price': current_price,
                    'vwap_level': df['vwap_lower_2'].iloc[-1]
                })
    
    # Real RSI signals
    if 'rsi' in df.columns and len(df) > 0:
        current_rsi = df['rsi'].iloc[-1]
        
        if current_rsi > 75:
            signals.append({
                'type': 'RSI_OVERBOUGHT',
                'direction': 'BEARISH',
                'strength': 'MEDIUM',
                'message': f'RSI severely overbought at {current_rsi:.1f}. Potential reversal zone.',
                'rsi_value': current_rsi
            })
        elif current_rsi < 25:
            signals.append({
                'type': 'RSI_OVERSOLD',
                'direction': 'BULLISH',
                'strength': 'MEDIUM',
                'message': f'RSI severely oversold at {current_rsi:.1f}. Potential bounce zone.',
                'rsi_value': current_rsi
            })
    
    # Real market depth signals
    if depth_data:
        for key, depth_info in depth_data.items():
            depth_imbalance = depth_info.get('depth_imbalance', 0)
            
            if abs(depth_imbalance) > 30:  # Significant real imbalance
                direction = "BULLISH" if depth_imbalance > 0 else "BEARISH"
                signals.append({
                    'type': 'DEPTH_IMBALANCE',
                    'direction': direction,
                    'strength': 'HIGH',
                    'message': f'Significant depth imbalance: {depth_imbalance:.1f}%. {direction.title()} pressure detected.',
                    'imbalance_value': depth_imbalance,
                    'bid_qty': depth_info.get('total_bid_qty', 0),
                    'ask_qty': depth_info.get('total_ask_qty', 0)
                })
    
    # Real options-based signals
    atm_data = options_data[options_data['Zone'] == 'ATM']
    if not atm_data.empty:
        atm_row = atm_data.iloc[0]
        
        # Real OI change analysis
        ce_oi_change = atm_row.get('CE_OI_Change', 0)
        pe_oi_change = atm_row.get('PE_OI_Change', 0)
        
        if abs(ce_oi_change) > 10000 or abs(pe_oi_change) > 10000:  # Significant OI change
            if pe_oi_change > ce_oi_change * 1.5:
                signals.append({
                    'type': 'STRONG_PUT_WRITING',
                    'direction': 'BULLISH',
                    'strength': 'HIGH',
                    'message': f'Strong PUT writing at ATM {atm_row["Strike"]}. PE OI change: +{pe_oi_change:,}',
                    'pe_oi_change': pe_oi_change,
                    'ce_oi_change': ce_oi_change
                })
            elif ce_oi_change > pe_oi_change * 1.5:
                signals.append({
                    'type': 'STRONG_CALL_WRITING',
                    'direction': 'BEARISH',
                    'strength': 'HIGH',
                    'message': f'Strong CALL writing at ATM {atm_row["Strike"]}. CE OI change: +{ce_oi_change:,}',
                    'ce_oi_change': ce_oi_change,
                    'pe_oi_change': pe_oi_change
                })
        
        # Real PCR analysis
        pcr_oi = atm_row.get('PCR_OI', 0)
        if pcr_oi > 2.0:
            signals.append({
                'type': 'EXTREME_PCR',
                'direction': 'BULLISH',
                'strength': 'MEDIUM',
                'message': f'Extreme PCR at {pcr_oi:.2f}. High put concentration suggests support.',
                'pcr_value': pcr_oi
            })
        elif pcr_oi < 0.5:
            signals.append({
                'type': 'EXTREME_PCR',
                'direction': 'BEARISH',
                'strength': 'MEDIUM',
                'message': f'Extreme PCR at {pcr_oi:.2f}. High call concentration suggests resistance.',
                'pcr_value': pcr_oi
            })
    
    # Real volume surge analysis
    if len(df) >= 20:
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        if current_volume > avg_volume * 2.5:  # Significant volume surge
            signals.append({
                'type': 'VOLUME_SURGE',
                'direction': 'NEUTRAL',
                'strength': 'HIGH',
                'message': f'Volume surge detected: {current_volume:,} vs avg {avg_volume:,.0f} ({current_volume/avg_volume:.1f}x)',
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': current_volume / avg_volume
            })
    
    return signals

def send_real_signals(signals, current_price):
    """Send only real signals via Telegram"""
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
        
        send_alert(message, priority)

def main():
    st.title("🏛️ Real Institutional Nifty Analyzer")
    st.markdown("*Using only real market data - No dummy calculations*")
    
    # Show market status
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"Market closed. Current time: {current_time.strftime('%H:%M:%S IST')}")
        st.info("Analysis using latest available real data")
    
    # Sidebar settings
    st.sidebar.header("📊 Real Data Settings")
    timeframe = st.sidebar.selectbox("Timeframe", ["1", "3", "5", "15"], index=2)
    enable_alerts = st.sidebar.checkbox("Real Signal Alerts", value=True)
    enable_depth = st.sidebar.checkbox("Market Depth Analysis", value=True)
    
    # Analysis options
    st.sidebar.subheader("🎯 Analysis Options")
    show_options = st.sidebar.checkbox("Options Analysis", value=True)
    show_depth_chart = st.sidebar.checkbox("Depth Visualization", value=True)
    
    # Initialize API
    api = RealInstitutionalAPI()
    
    # Main layout
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        st.header("📈 Real Price Analysis")
        
        # Get real data
        data = api.get_intraday_data(timeframe)
        df = process_candle_data(data) if data else pd.DataFrame()
        
        # Get current price from LTP API
        ltp_data = api.get_ltp_data()
        current_price = None
        if ltp_data and 'data' in ltp_data:
            for exchange, instruments in ltp_data['data'].items():
                for scrip_id, price_data in instruments.items():
                    current_price = price_data.get('last_price', 0)
                    break
        
        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]
        
        # Display real metrics
        if not df.empty and current_price:
            # Get OHLC data
            ohlc_data = api.get_ohlc_data({NIFTY_SEG: [NIFTY_SCRIP]})
            day_high = day_low = day_open = 0
            
            if ohlc_data and 'data' in ohlc_data:
                for exchange, instruments in ohlc_data['data'].items():
                    for scrip_id, ohlc_info in instruments.items():
                        ohlc = ohlc_info.get('ohlc', {})
                        day_high = ohlc.get('high', 0)
                        day_low = ohlc.get('low', 0)
                        day_open = ohlc.get('open', 0)
                        break
            
            # Calculate real indicators
            df_enhanced = calculate_real_technical_indicators(df)
            current_vwap = df_enhanced['vwap'].iloc[-1] if 'vwap' in df_enhanced.columns else current_price
            current_rsi = df_enhanced['rsi'].iloc[-1] if 'rsi' in df_enhanced.columns else 50
            current_atr = df_enhanced['atr'].iloc[-1] if 'atr' in df_enhanced.columns else 0
            
            # Real metrics display
            col1_m, col2_m, col3_m, col4_m = st.columns(4)
            with col1_m:
                change = current_price - day_open if day_open > 0 else 0
                change_pct = (change / day_open * 100) if day_open > 0 else 0
                st.metric("Nifty", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            
            with col2_m:
                st.metric("Day High", f"₹{day_high:.2f}" if day_high > 0 else "N/A")
            
            with col3_m:
                st.metric("Day Low", f"₹{day_low:.2f}" if day_low > 0 else "N/A")
            
            with col4_m:
                vwap_dev = ((current_price - current_vwap) / current_vwap * 100) if current_vwap > 0 else 0
                st.metric("VWAP Dev", f"{vwap_dev:+.2f}%")
            
            # Additional real metrics
            col5_m, col6_m, col7_m, col8_m = st.columns(4)
            with col5_m:
                st.metric("RSI", f"{current_rsi:.1f}")
            with col6_m:
                st.metric("ATR", f"₹{current_atr:.2f}")
            with col7_m:
                volume_latest = df['volume'].iloc[-1] if not df.empty else 0
                st.metric("Latest Vol", f"{volume_latest:,}")
            with col8_m:
                avg_vol = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else volume_latest
                vol_ratio = volume_latest / avg_vol if avg_vol > 0 else 1
                st.metric("Vol Ratio", f"{vol_ratio:.1f}x")
        
        # Create real price chart
        if not df.empty:
            fig = create_real_price_chart(df, f"Real Price Analysis - {timeframe}min")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No price data available")
    
    with col2:
        st.header("📊 Real Market Data")
        
        # Market depth analysis
        if enable_depth and current_price:
            st.subheader("Real Market Depth")
            
            depth_instruments = {NIFTY_SEG: [NIFTY_SCRIP]}
            depth_data = api.get_market_depth(depth_instruments)
            
            if depth_data:
                processed_depth = process_real_market_depth(depth_data)
                if processed_depth and show_depth_chart:
                    depth_fig = create_real_market_depth_chart(processed_depth)
                    st.plotly_chart(depth_fig, use_container_width=True)
                
                # Display real depth metrics
                if processed_depth:
                    first_key = list(processed_depth.keys())[0]
                    depth_info = processed_depth[first_key]
                    
                    st.write("**Real Depth Metrics:**")
                    st.write(f"Depth Imbalance: {depth_info.get('depth_imbalance', 0):.1f}%")
                    st.write(f"Bid-Ask Spread: ₹{depth_info.get('bid_ask_spread', 0):.2f}")
                    st.write(f"Total Bid Qty: {depth_info.get('total_bid_qty', 0):,}")
                    st.write(f"Total Ask Qty: {depth_info.get('total_ask_qty', 0):,}")
            else:
                st.info("Market depth data not available")
        
        # Real options analysis
        if show_options:
            st.subheader("Real Options Data")
            
            expiry_data = get_expiry_list()
            if expiry_data and 'data' in expiry_data:
                expiry_dates = expiry_data['data']
                selected_expiry = st.selectbox("Expiry", expiry_dates)
                
                underlying_price, options_summary, summary_metrics = analyze_real_options(selected_expiry, current_price)
                
                if underlying_price and options_summary is not None:
                    st.write(f"**Underlying**: ₹{underlying_price:.2f}")
                    st.write(f"**Days to Expiry**: {summary_metrics.get('days_to_expiry', 0)}")
                    st.write(f"**PCR (OI)**: {summary_metrics.get('pcr_oi', 0):.3f}")
                    st.write(f"**PCR (Volume)**: {summary_metrics.get('pcr_volume', 0):.3f}")
                    
                    # Display key options data
                    display_cols = ['Strike', 'Zone', 'Level', 'CE_OI', 'PE_OI', 'PCR_OI', 'CE_IV', 'PE_IV']
                    st.dataframe(options_summary[display_cols], use_container_width=True, height=300)
                    
                    # Generate real signals
                    if enable_alerts and is_market_hours():
                        depth_data_for_signals = None
                        if enable_depth:
                            depth_instruments = {NIFTY_SEG: [NIFTY_SCRIP]}
                            depth_raw = api.get_market_depth(depth_instruments)
                            if depth_raw:
                                depth_data_for_signals = process_real_market_depth(depth_raw)
                        
                        signals = detect_real_signals(df, options_summary, summary_metrics, current_price, depth_data_for_signals)
                        if signals:
                            send_real_signals(signals, current_price)
                            st.success(f"🔔 {len(signals)} real signal(s) detected and sent!")
                            
                            # Show recent signals
                            for signal in signals[-2:]:
                                signal_color = "🔴" if signal['strength'] == 'HIGH' else "🟡"
                                st.info(f"{signal_color} {signal['type']}: {signal['message'][:80]}...")
                else:
                    st.error("Options data not available")
            else:
                st.error("Expiry data not available")
    
    # Real options chart
    if show_options and 'options_summary' in locals() and options_summary is not None:
        st.header("📊 Real Options Analysis")
        options_fig = create_real_options_chart(options_summary)
        st.plotly_chart(options_fig, use_container_width=True)
    
    # Footer
    current_time_str = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"🔄 Updated: {current_time_str}")
    st.sidebar.write("**Data Sources**: All real-time from Dhan API")
    st.sidebar.write("**No Dummy Data**: All calculations use actual market data")
    
    if st.sidebar.button("📨 Test Alert"):
        send_alert("Test message from Real Institutional Analyzer", "NORMAL")
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()