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

# Page config
st.set_page_config(page_title="Advanced Nifty Analyzer", page_icon="📈", layout="wide")

# Market hours check
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5: return False
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return market_start <= now <= market_end

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

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap_bands(df):
    if df.empty:
        return df
    
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cum_vol'] = df['volume'].cumsum()
    df['cum_tp_vol'] = (df['typical_price'] * df['volume']).cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    df['sq_dev'] = (df['typical_price'] - df['vwap']) ** 2
    df['cum_sq_dev_vol'] = (df['sq_dev'] * df['volume']).cumsum()
    df['vwap_std'] = np.sqrt(df['cum_sq_dev_vol'] / df['cum_vol'])
    df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
    df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
    df['vwap_upper_2'] = df['vwap'] + 2 * df['vwap_std']
    df['vwap_lower_2'] = df['vwap'] - 2 * df['vwap_std']
    
    return df

def calculate_atr(df, period=14):
    if df.empty or len(df) < period:
        return df
    
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['l_pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    return df

def calculate_implied_volatility(option_price, spot_price, strike_price, time_to_expiry, option_type='CE', risk_free_rate=0.07):
    try:
        if time_to_expiry <= 0 or option_price <= 0:
            return 0
        
        iv = 0.3
        
        for _ in range(50):
            d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * iv**2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
            d2 = d1 - iv * np.sqrt(time_to_expiry)
            
            if option_type == 'CE':
                theoretical_price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                theoretical_price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
            
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
            
            if abs(theoretical_price - option_price) < 0.001 or vega == 0:
                break
                
            iv = iv - (theoretical_price - option_price) / vega
            iv = max(0.001, min(5.0, iv))
            
        return iv
    except:
        return 0

def calculate_gamma(spot_price, strike_price, time_to_expiry, iv, risk_free_rate=0.07):
    try:
        if time_to_expiry <= 0 or iv <= 0:
            return 0
        
        d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * iv**2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
        gamma = norm.pdf(d1) / (spot_price * iv * np.sqrt(time_to_expiry))
        return gamma
    except:
        return 0

def analyze_volume_spread(df_options, spot_price):
    results = []
    
    for _, row in df_options.iterrows():
        ce_vol = row.get('volume_CE', 0)
        pe_vol = row.get('volume_PE', 0)
        
        ce_bid_qty = row.get('top_bid_quantity_CE', 0)
        ce_ask_qty = row.get('top_ask_quantity_CE', 0)
        pe_bid_qty = row.get('top_bid_quantity_PE', 0)
        pe_ask_qty = row.get('top_ask_quantity_PE', 0)
        
        ce_buy_vol = ce_vol * 0.6 if (ce_bid_qty > ce_ask_qty) else ce_vol * 0.4
        ce_sell_vol = ce_vol - ce_buy_vol
        
        pe_buy_vol = pe_vol * 0.6 if (pe_bid_qty > pe_ask_qty) else pe_vol * 0.4
        pe_sell_vol = pe_vol - pe_buy_vol
        
        ce_delta = ce_buy_vol - ce_sell_vol
        pe_delta = pe_buy_vol - pe_sell_vol
        
        ce_delta_bias = "Bullish" if ce_delta > 0 else "Bearish" if ce_delta < 0 else "Neutral"
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

def calculate_volume_profile(df, bins=20):
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    
    profiles = []
    
    for idx, row in df.iterrows():
        high = row['high']
        low = row['low']
        volume = row['volume']
        close = row['close']
        
        if high == low:
            continue
            
        bins_edges = np.linspace(low, high, bins + 1)
        bins_center = (bins_edges[:-1] + bins_edges[1:]) / 2
        vol_bins = np.zeros(bins)
        
        close_bin = np.digitize(close, bins_edges) - 1
        close_bin = max(0, min(bins - 1, close_bin))
        
        for i in range(bins):
            if i == close_bin:
                vol_bins[i] = volume * 0.4
            else:
                distance = abs(bins_center[i] - close)
                max_distance = max(abs(bins_center - close))
                if max_distance > 0:
                    weight = 1 - (distance / max_distance)
                    vol_bins[i] = volume * 0.6 * weight / (bins - 1)
        
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

def create_enhanced_chart(df, title, interval):
    if df.empty:
        return go.Figure()
    
    df = calculate_vwap_bands(df)
    df = calculate_atr(df)
    df['rsi'] = calculate_rsi(df)
    
    volume_profile_df = calculate_volume_profile(df)
    
    fig = make_subplots(
        rows=5, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=('Price with VWAP Bands', 'Volume Profile', 'Volume', 'RSI', 'ATR')
    )
    
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    if 'vwap' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap'], name='VWAP', line=dict(color='#ffff00', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_upper_1'], name='VWAP +1σ', line=dict(color='#ff9900', width=1, dash='dash'), opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_lower_1'], name='VWAP -1σ', line=dict(color='#ff9900', width=1, dash='dash'), opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_upper_2'], name='VWAP +2σ', line=dict(color='#ff0000', width=1, dash='dot'), opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap_lower_2'], name='VWAP -2σ', line=dict(color='#ff0000', width=1, dash='dot'), opacity=0.5), row=1, col=1)
    
    if len(df) > 50:
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", '#4444ff']
        
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            x_start, x_end = df['datetime'].min(), df['datetime'].max()
            
            for pivot in pivots[-5:]:
                fig.add_shape(type="line", x0=x_start, x1=x_end, y0=pivot['value'], y1=pivot['value'], line=dict(color=color, width=1, dash="dash"), row=1, col=1)
    
    if not volume_profile_df.empty:
        fig.add_trace(go.Scatter(x=volume_profile_df['datetime'], y=volume_profile_df['poc'], mode='lines', name='POC (Point of Control)', line=dict(color='#0066ff', width=2, dash='dash')), row=2, col=1)
    
    volume_colors = ['#00ff88' if close >= open else '#ff4444' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], name='Volume', marker_color=volume_colors, opacity=0.7), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['rsi'], name='RSI', line=dict(color='#ff9900', width=2)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=4, col=1)
    
    if 'atr' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['atr'], name='ATR', line=dict(color='#00ffff', width=2)), row=5, col=1)
    
    fig.update_layout(title=title, template='plotly_dark', height=1200, xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)
    fig.update_yaxes(title_text="ATR", row=5, col=1)
    fig.update_yaxes(title_text="Volume Profile", row=2, col=1)
    
    return fig

def analyze_advanced_options(expiry, spot_price):
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
    
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df_filtered = df[abs(df['strikePrice'] - atm_strike) <= 100]
    
    if df_filtered.empty:
        return None, None, None, None, None
    
    try:
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        expiry_datetime = expiry_date.replace(hour=15, minute=30)
        current_time = datetime.now()
        time_to_expiry = max((expiry_datetime - current_time).total_seconds() / (365.25 * 24 * 3600), 0.001)
    except:
        time_to_expiry = 0.027
    
    iv_results = []
    gamma_results = []
    
    for _, row in df_filtered.iterrows():
        strike = row['strikePrice']
        
        ce_price = row.get('last_price_CE', 0)
        pe_price = row.get('last_price_PE', 0)
        
        ce_iv = calculate_implied_volatility(ce_price, underlying, strike, time_to_expiry, 'CE')
        pe_iv = calculate_implied_volatility(pe_price, underlying, strike, time_to_expiry, 'PE')
        
        ce_gamma = calculate_gamma(underlying, strike, time_to_expiry, ce_iv)
        pe_gamma = calculate_gamma(underlying, strike, time_to_expiry, pe_iv)
        
        if ce_iv > 0 and pe_iv > 0:
            iv_diff_pct = abs(ce_iv - pe_iv) / max(ce_iv, pe_iv) * 100
            
            if iv_diff_pct > 5:
                ce_iv_bias = "Cheaper" if ce_iv < pe_iv else "Expensive"
                pe_iv_bias = "Cheaper" if pe_iv < ce_iv else "Expensive"
            else:
                ce_iv_bias = "Neutral"
                pe_iv_bias = "Neutral"
        else:
            ce_iv_bias = "N/A"
            pe_iv_bias = "N/A"
        
        ce_oi = row.get('oi_CE', 0)
        pe_oi = row.get('oi_PE', 0)
        
        ce_gex = ce_gamma * ce_oi * underlying * 0.01
        pe_gex = pe_gamma * pe_oi * underlying * 0.01
        net_gex = ce_gex - pe_gex
        
        if abs(ce_gex) > 0 and abs(pe_gex) > 0:
            gamma_diff_pct = abs(abs(ce_gex) - abs(pe_gex)) / max(abs(ce_gex), abs(pe_gex)) * 100
            
            if gamma_diff_pct > 20:
                gamma_bias = "CE Higher Gamma" if abs(ce_gex) > abs(pe_gex) else "PE Higher Gamma"
            else:
                gamma_bias = "Neutral"
        else:
            gamma_bias = "N/A"
        
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
    
    df_filtered['Zone'] = df_filtered['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
    
    bias_results = []
    for i, row in df_filtered.iterrows():
        strike = row['strikePrice']
        
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
        
        ce_delta = row.get('greeks', {}).get('delta', 0) if 'greeks' in row else 0
        pe_delta = row.get('greeks', {}).get('delta', 0) if 'greeks' in row else 0
        
        ce_delta_bias = "Bullish" if ce_delta > 0.5 else "Bearish" if ce_delta < 0.3 else "Neutral"
        pe_delta_bias = "Bearish" if pe_delta < -0.5 else "Bullish" if pe_delta > -0.3 else "Neutral"
        
        ce_price = row.get('last_price_CE', 0)
        pe_price = row.get('last_price_PE', 0)
        
        ce_iv = calculate_implied_volatility(ce_price, underlying, strike, time_to_expiry, 'CE')
        pe_iv = calculate_implied_volatility(pe_price, underlying, strike, time_to_expiry, 'PE')
        
        if ce_iv > 0 and pe_iv > 0:
            iv_diff_pct = abs(ce_iv - pe_iv) / max(ce_iv, pe_iv) * 100
            
            if iv_diff_pct > 5:
                if ce_iv < pe_iv:
                    iv_bias = "CE Cheaper"
                else:
                    iv_bias = "PE Cheaper"
            else:
                iv_bias = "Neutral"
        else:
            iv_bias = "N/A"
        
        ce_gamma = calculate_gamma(underlying, strike, time_to_expiry, ce_iv)
        pe_gamma = calculate_gamma(underlying, strike, time_to_expiry, pe_iv)
        
        ce_gex = ce_gamma * curr_oi_ce * underlying * 0.01
        pe_gex = pe_gamma * curr_oi_pe * underlying * 0.01
        
        if abs(ce_gex) > 0 and abs(pe_gex) > 0:
            gamma_diff_pct = abs(abs(ce_gex) - abs(pe_gex)) / max(abs(ce_gex), abs(pe_gex)) * 100
            
            if gamma_diff_pct > 20:
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
            "Net_GEX": round(ce_gex - pe_gex, 0),
            "changeinOpenInterest_CE": chg_oi_ce,
            "changeinOpenInterest_PE": chg_oi_pe
        })
    
    volume_spread_df = analyze_volume_spread(df_filtered, underlying)
    
    return underlying, pd.DataFrame(bias_results), pd.DataFrame(iv_results), pd.DataFrame(gamma_results), volume_spread_df

def calculate_final_bias_score(df, option_summary, iv_df, gamma_df, volume_spread_df, current_price):
    if df.empty or option_summary is None:
        return "Insufficient Data", 0, {}
    
    scores = {}
    
    # 1. RSI Score (-2 to +2)
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
    
    # Get ATM data for remaining scores
    atm_data = option_summary[option_summary['Zone'] == 'ATM']
    if atm_data.empty:
        return "Insufficient Data", 0, {}
    
    atm_row = atm_data.iloc[0]
    
    # 2. ATM Change in OI Score (-1 to +1)
    chg_oi_score = 1 if atm_row['ChgOI_Bias'] == 'Bullish' else -1 if atm_row['ChgOI_Bias'] == 'Bearish' else 0
    scores['ChgOI'] = chg_oi_score
    
    # 3. ATM Volume Score (-1 to +1)
    volume_score = 1 if atm_row['Volume_Bias'] == 'Bullish' else -1 if atm_row['Volume_Bias'] == 'Bearish' else 0
    scores['Volume'] = volume_score
    
    # 4. ATM CE Delta Score (-1 to +1)
    ce_delta_score = 1 if atm_row['CE_Delta_Bias'] == 'Bullish' else -1 if atm_row['CE_Delta_Bias'] == 'Bearish' else 0
    scores['CE_Delta'] = ce_delta_score
    
    # 5. ATM PE Delta Score (-1 to +1)
    pe_delta_score = 1 if atm_row['PE_Delta_Bias'] == 'Bullish' else -1 if atm_row['PE_Delta_Bias'] == 'Bearish' else 0
    scores['PE_Delta'] = pe_delta_score
    
    # 6. ATM IV Score (-1 to +1)
    if atm_row['IV_Bias'] == 'CE Cheaper':
        iv_score = 1  # Calls cheaper = Bullish
    elif atm_row['IV_Bias'] == 'PE Cheaper':
        iv_score = -1  # Puts cheaper = Bearish
    else:
        iv_score = 0
    scores['IV'] = iv_score
    
    # 7. ATM Gamma Score (-1 to +1)
    # PE High Gamma = Bullish, CE High Gamma = Bearish
    gamma_score = 1 if atm_row['Gamma_Bias'] == 'Bullish' else -1 if atm_row['Gamma_Bias'] == 'Bearish' else 0
    scores['Gamma'] = gamma_score
    
    # Calculate total score
    total_score = sum(scores.values())
    max_possible = 9  # RSI(2) + 6 other components(1 each)
    
    # Determine bias
    if total_score >= 4:
        bias = "Strong Bullish"
    elif total_score >= 2:
        bias = "Bullish"
    elif total_score <= -4:
        bias = "Strong Bearish"
    elif total_score <= -2:
        bias = "Bearish"
    else:
        bias = "Neutral"
    
    # Market dynamics based on gamma
    if atm_row['Gamma_Bias'] == 'Bullish':
        market_dynamics = "PE Gamma Dominant (Bullish)"
    elif atm_row['Gamma_Bias'] == 'Bearish':
        market_dynamics = "CE Gamma Dominant (Bearish)"
    else:
        market_dynamics = "Balanced Gamma"
    
    return bias, total_score, scores, market_dynamics

def check_advanced_signals(df, option_data, iv_df, gamma_df, volume_spread_df, current_price, proximity=5):
    if df.empty or option_data is None or not current_price:
        return
    
    df['rsi'] = calculate_rsi(df)
    df = calculate_vwap_bands(df)
    df = calculate_atr(df)
    
    current_rsi = df['rsi'].iloc[-1] if not df.empty else None
    current_vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns else None
    current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
    
    bias, score, detailed_scores, dynamics = calculate_final_bias_score(df, option_data, iv_df, gamma_df, volume_spread_df, current_price)
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty:
        return
    
    row = atm_data.iloc[0]
    
    strong_bullish = (
        bias == "Strong Bullish" and
        current_rsi is not None and current_rsi < 70 and
        current_vwap is not None and current_price > current_vwap and
        row['CE_Delta_Bias'] == 'Bullish' and
        row['IV_Bias'] in ['CE Cheaper', 'Neutral']
    )
    
    strong_bearish = (
        bias == "Strong Bearish" and
        current_rsi is not None and current_rsi > 30 and
        current_vwap is not None and current_price < current_vwap and
        row['PE_Delta_Bias'] == 'Bearish' and
        row['IV_Bias'] in ['PE Cheaper', 'Neutral']
    )
    
    if strong_bullish or strong_bearish:
        signal_type = "STRONG CALL" if strong_bullish else "STRONG PUT"
        
        message = f"""
🔥 MASTER {signal_type} SIGNAL 🔥

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}
📊 Bias Score: {score}/9 ({bias})

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
            
            df_calc = calculate_vwap_bands(df.copy())
            df_calc = calculate_atr(df_calc)
            df_calc['rsi'] = calculate_rsi(df_calc)
            
            current_rsi = df_calc['rsi'].iloc[-1] if not df_calc.empty else None
            current_vwap = df_calc['vwap'].iloc[-1] if 'vwap' in df_calc.columns else None
            current_atr = df_calc['atr'].iloc[-1] if 'atr' in df_calc.columns else None
            
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
            fig = create_enhanced_chart(df, f"Nifty {interval}min", interval)
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
                    
                    bias, score, detailed_scores, dynamics = calculate_final_bias_score(df, option_summary, iv_df, gamma_df, volume_spread_df, current_price)
                    
                    st.subheader("🎪 Final Bias Dashboard")
                    bias_col1, bias_col2 = st.columns(2)
                    
                    with bias_col1:
                        bias_color = "green" if "Bullish" in bias else "red" if "Bearish" in bias else "gray"
                        st.markdown(f"**Market Bias:** <span style='color:{bias_color}'>{bias}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Score:** {score}/9")
                    
                    with bias_col2:
                        st.markdown(f"**Dynamics:** {dynamics}")
                        if current_atr:
                            stop_loss = round(current_atr * 1.5, 0)
                            st.markdown(f"**Suggested SL:** ±{stop_loss} pts")
                    
                    st.write("**Component Scores:**")
                    score_df = pd.DataFrame([detailed_scores])
                    st.dataframe(score_df, use_container_width=True)
                    
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
                        
                        st.write("**Trading Recommendations:**")
                        if not option_summary.empty:
                            atm_row = option_summary[option_summary['Zone'] == 'ATM'].iloc[0]
                            if atm_row['IV_Bias'] == 'CE Cheaper':
                                st.write("• Consider Call options (cheaper IV)")
                            elif atm_row['IV_Bias'] == 'PE Cheaper':
                                st.write("• Consider Put options (cheaper IV)")
                            
                            if atm_row['Gamma_Bias'] == 'Bullish':
                                st.write("• PE Higher Gamma - Bullish gamma bias")
                            elif atm_row['Gamma_Bias'] == 'Bearish':
                                st.write("• CE Higher Gamma - Bearish gamma bias")
                            elif atm_row['Gamma_Bias'] == 'Neutral':
                                st.write("• Balanced gamma exposure")
                    
                    if enable_signals and not df.empty and is_market_hours():
                        with st.spinner("Checking for signals..."):
                            check_advanced_signals(df, option_summary, iv_df, gamma_df, volume_spread_df, current_price, proximity)
                else:
                    st.error("Options data unavailable")
            else:
                st.error("Current price not available")
        else:
            st.error("Expiry data unavailable")
    
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time}")
    
    if st.sidebar.button("🧪 Test Telegram"):
        send_telegram("🔔 Test message from Advanced Nifty Analyzer")
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()
