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
    st_autorefresh(interval=80000, key="refresh")
else:
    st.info("Market is closed. Auto-refresh disabled.")

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))
NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

# Volume Profile Configuration
VP_CONFIG = {
    "BINS": 20,
    "TIMEFRAME": "30T",  # 30 minutes for intraday analysis
    "POC_COLOR": "#FFD700",  # Gold color for POC
    "VOLUME_COLOR": "#1f77b4",
    "HIGH_VOLUME_COLOR": "#ff7f0e"
}

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

def calculate_volume_profile(df, timeframe="30T", bins=20):
    """
    Calculate Volume Profile / Money Flow indicator
    """
    if df.empty or len(df) < 10:
        return []
    
    # Resample data to higher timeframe
    df_temp = df.set_index('datetime')
    try:
        ohlc = df_temp.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        if len(ohlc) == 0:
            return []
        
        profiles = []
        
        for idx, row in ohlc.iterrows():
            high, low = row["high"], row["low"]
            if high == low:  # Skip if no price movement
                continue
                
            step = (high - low) / bins
            price_bins = np.linspace(low, high, bins + 1)
            
            # Initialize volume histogram
            volume_hist = np.zeros(bins)
            
            # Get data for this timeframe period
            period_end = idx + pd.Timedelta(timeframe)
            mask = (df_temp.index >= idx) & (df_temp.index < period_end)
            sub_df = df_temp.loc[mask]
            
            if len(sub_df) == 0:
                continue
            
            # Distribute volume across price levels
            for _, r in sub_df.iterrows():
                price = r["close"]
                volume = r["volume"]
                
                # Find which bin this price belongs to
                for i in range(bins):
                    if price >= price_bins[i] and price < price_bins[i + 1]:
                        volume_hist[i] += volume
                        break
                    elif i == bins - 1 and price >= price_bins[i]:  # Last bin edge case
                        volume_hist[i] += volume
                        break
            
            # Calculate Point of Control (POC) - price level with highest volume
            if volume_hist.max() > 0:
                poc_idx = np.argmax(volume_hist)
                poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
                
                # Calculate Value Area (70% of total volume)
                total_volume = volume_hist.sum()
                if total_volume > 0:
                    # Sort bins by volume to find value area
                    sorted_indices = np.argsort(volume_hist)[::-1]
                    cumulative_volume = 0
                    value_area_indices = []
                    
                    for bin_idx in sorted_indices:
                        cumulative_volume += volume_hist[bin_idx]
                        value_area_indices.append(bin_idx)
                        if cumulative_volume >= 0.7 * total_volume:
                            break
                    
                    # Calculate Value Area High and Low
                    if value_area_indices:
                        va_high = max([price_bins[i + 1] for i in value_area_indices])
                        va_low = min([price_bins[i] for i in value_area_indices])
                    else:
                        va_high = high
                        va_low = low
                
                profiles.append({
                    "time": idx,
                    "high": high,
                    "low": low,
                    "poc": poc_price,
                    "va_high": va_high,
                    "va_low": va_low,
                    "volume_hist": volume_hist,
                    "price_bins": price_bins,
                    "total_volume": total_volume
                })
        
        return profiles
        
    except Exception as e:
        st.error(f"Error calculating volume profile: {e}")
        return []

def get_volume_profile_insights(profiles, current_price):
    """
    Analyze volume profile and provide trading insights
    """
    if not profiles or current_price is None:
        return {}
    
    # Get the latest profile
    latest_profile = profiles[-1] if profiles else None
    if not latest_profile:
        return {}
    
    insights = {
        "poc": latest_profile["poc"],
        "va_high": latest_profile["va_high"],
        "va_low": latest_profile["va_low"],
        "total_volume": latest_profile["total_volume"],
        "price_vs_poc": "",
        "price_vs_va": "",
        "bias": "NEUTRAL",
        "strength": "WEAK"
    }
    
    # Analyze current price vs POC
    poc_diff = current_price - insights["poc"]
    poc_diff_pct = (poc_diff / insights["poc"]) * 100 if insights["poc"] > 0 else 0
    
    if abs(poc_diff_pct) < 0.1:
        insights["price_vs_poc"] = f"AT POC (±{poc_diff:+.1f})"
        insights["bias"] = "NEUTRAL"
    elif poc_diff > 0:
        insights["price_vs_poc"] = f"ABOVE POC (+{poc_diff:.1f}, +{poc_diff_pct:.2f}%)"
        insights["bias"] = "BULLISH"
    else:
        insights["price_vs_poc"] = f"BELOW POC ({poc_diff:.1f}, {poc_diff_pct:.2f}%)"
        insights["bias"] = "BEARISH"
    
    # Analyze current price vs Value Area
    if current_price > insights["va_high"]:
        insights["price_vs_va"] = f"ABOVE VA (+{current_price - insights['va_high']:.1f})"
        if insights["bias"] == "BULLISH":
            insights["strength"] = "STRONG"
    elif current_price < insights["va_low"]:
        insights["price_vs_va"] = f"BELOW VA (-{insights['va_low'] - current_price:.1f})"
        if insights["bias"] == "BEARISH":
            insights["strength"] = "STRONG"
    else:
        insights["price_vs_va"] = "INSIDE VALUE AREA"
        insights["strength"] = "MODERATE" if insights["bias"] != "NEUTRAL" else "WEAK"
    
    return insights

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

def create_chart(df, title):
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, 
                       row_heights=[0.6, 0.2, 0.2],
                       subplot_titles=('Price', 'Volume', 'Volume Profile'))
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Volume bars
    volume_colors = ['#00ff88' if close >= open else '#ff4444' 
                    for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)
    
    # Calculate and add Volume Profile
    if len(df) > 50:
        profiles = calculate_volume_profile(df, VP_CONFIG["TIMEFRAME"], VP_CONFIG["BINS"])
        
        # Add Volume Profile visualization
        for profile in profiles[-5:]:  # Show last 5 profiles
            time_start = profile["time"]
            time_end = time_start + pd.Timedelta(VP_CONFIG["TIMEFRAME"])
            
            # Add POC line
            fig.add_shape(
                type="line",
                x0=time_start, x1=time_end,
                y0=profile["poc"], y1=profile["poc"],
                line=dict(color=VP_CONFIG["POC_COLOR"], width=2, dash="dash"),
                row=1, col=1
            )
            
            # Add Value Area
            fig.add_shape(
                type="rect",
                x0=time_start, x1=time_end,
                y0=profile["va_low"], y1=profile["va_high"],
                fillcolor="rgba(255, 215, 0, 0.1)",
                line=dict(color="rgba(255, 215, 0, 0.3)", width=1),
                row=1, col=1
            )
            
            # Add volume histogram in third subplot
            price_centers = (profile["price_bins"][:-1] + profile["price_bins"][1:]) / 2
            max_vol = profile["volume_hist"].max()
            if max_vol > 0:
                # Normalize volume for display
                normalized_vol = profile["volume_hist"] / max_vol * 100
                
                fig.add_trace(go.Bar(
                    x=[time_start] * len(price_centers),
                    y=normalized_vol,
                    name=f'VP {time_start.strftime("%H:%M")}',
                    orientation='v',
                    marker_color=VP_CONFIG["VOLUME_COLOR"],
                    opacity=0.6,
                    showlegend=False
                ), row=3, col=1)
        
        # Add pivot lines
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", "#4444ff"]
        
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            x_start, x_end = df['datetime'].min(), df['datetime'].max()
            
            for pivot in pivots[-5:]:
                fig.add_shape(type="line", x0=x_start, x1=x_end,
                            y0=pivot['value'], y1=pivot['value'],
                            line=dict(color=color, width=1, dash="dash"), row=1, col=1)
    
    fig.update_layout(title=title, template='plotly_dark', height=800,
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
    
    # Calculate Volume Profile insights
    profiles = calculate_volume_profile(df, VP_CONFIG["TIMEFRAME"], VP_CONFIG["BINS"])
    vp_insights = get_volume_profile_insights(profiles, current_price)
    
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
    pivots = get_pivots(df, "5") + get_pivots(df, "10") + get_pivots(df, "15")
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
        
        # Add Volume Profile confirmation
        vp_confirmation = ""
        if vp_insights:
            if primary_bullish_signal and vp_insights["bias"] == "BULLISH":
                vp_confirmation = f"✅ VP CONFIRMS: {vp_insights['price_vs_poc']} | {vp_insights['strength']} {vp_insights['bias']}"
            elif primary_bearish_signal and vp_insights["bias"] == "BEARISH":
                vp_confirmation = f"✅ VP CONFIRMS: {vp_insights['price_vs_poc']} | {vp_insights['strength']} {vp_insights['bias']}"
            else:
                vp_confirmation = f"⚠️ VP MIXED: {vp_insights['price_vs_poc']} | {vp_insights['bias']} bias"
        
        if primary_bullish_signal or primary_bearish_signal:
            signal_type = "CALL" if primary_bullish_signal else "PUT"
            price_diff = current_price - pivot_level['value']
            
            message = f"""
🚨 PRIMARY NIFTY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M at ₹{pivot_level['value']:.2f}
🎯 ATM: {row['Strike']}

💰 VOLUME PROFILE ANALYSIS:
{vp_confirmation}
📊 POC: ₹{vp_insights.get('poc', 0):.1f} | VA: ₹{vp_insights.get('va_low', 0):.1f}-₹{vp_insights.get('va_high', 0):.1f}
🔄 Price vs VA: {vp_insights.get('price_vs_va', 'N/A')}

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
        
        # Add Volume Profile insights
        vp_info = ""
        if vp_insights:
            vp_info = f"""
💰 VOLUME PROFILE:
📊 {vp_insights.get('price_vs_poc', 'N/A')} | {vp_insights.get('price_vs_va', 'N/A')}
🎯 Bias: {vp_insights.get('strength', 'WEAK')} {vp_insights.get('bias', 'NEUTRAL')}
"""
        
        message = f"""
⚡ SECONDARY NIFTY {signal_type} SIGNAL - OI DOMINANCE ⚡

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']}

🔥 OI Dominance: {'PUT' if secondary_bullish_signal else 'CALL'} ChgOI {dominance_ratio:.1f}x higher
📊 All Bias Aligned: {row['ChgOI_Bias']}, {row['Volume_Bias']}, {row['Ask_Bias']}, {row['Bid_Bias']}
{vp_info}
ChgOI: CE {ce_chg_oi:,} | PE {pe_chg_oi:,}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success(f"⚡ SECONDARY {signal_type} signal sent!")

def main():
    st.title("📈 Nifty Trading Analyzer with Volume Profile")
    
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
    
    # Volume Profile Settings
    st.sidebar.subheader("Volume Profile")
    vp_timeframe = st.sidebar.selectbox("VP Timeframe", ["15T", "30T", "60T"], index=1)
    VP_CONFIG["TIMEFRAME"] = vp_timeframe
    VP_CONFIG["BINS"] = st.sidebar.slider("VP Bins", 10, 30, 20)
    
    api = DhanAPI()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chart with Volume Profile")
        
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
        
        # Display metrics and Volume Profile insights
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
            
            # Calculate and display Volume Profile insights
            if len(df) > 50:
                profiles = calculate_volume_profile(df, VP_CONFIG["TIMEFRAME"], VP_CONFIG["BINS"])
                vp_insights = get_volume_profile_insights(profiles, current_price)
                
                if vp_insights:
                    st.subheader("💰 Volume Profile Analysis")
                    col_vp1, col_vp2, col_vp3, col_vp4 = st.columns(4)
                    
                    with col_vp1:
                        st.metric("POC", f"₹{vp_insights['poc']:,.1f}")
                    with col_vp2:
                        st.metric("VA High", f"₹{vp_insights['va_high']:,.1f}")
                    with col_vp3:
                        st.metric("VA Low", f"₹{vp_insights['va_low']:,.1f}")
                    with col_vp4:
                        bias_color = "🟢" if vp_insights['bias'] == 'BULLISH' else "🔴" if vp_insights['bias'] == 'BEARISH' else "⚪"
                        st.metric("VP Bias", f"{bias_color} {vp_insights['bias']}")
                    
                    # Display detailed insights
                    st.info(f"📍 **Price Position**: {vp_insights['price_vs_poc']} | {vp_insights['price_vs_va']}")
                    st.info(f"💪 **Signal Strength**: {vp_insights['strength']} | **Total Volume**: {vp_insights['total_volume']:,.0f}")
        
        if not df.empty:
            fig = create_chart(df, f"Nifty {interval}min with Volume Profile")
            st.plotly_chart(fig, use_container_width=True)
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
        
        # Volume Profile Summary for Options Analysis
        if not df.empty and len(df) > 50:
            st.subheader("📊 VP Trading Levels")
            profiles = calculate_volume_profile(df, VP_CONFIG["TIMEFRAME"], VP_CONFIG["BINS"])
            vp_insights = get_volume_profile_insights(profiles, current_price)
            
            if vp_insights:
                # Create a summary table of key levels
                levels_data = {
                    "Level": ["POC", "VA High", "VA Low"],
                    "Price": [f"₹{vp_insights['poc']:,.1f}", 
                             f"₹{vp_insights['va_high']:,.1f}", 
                             f"₹{vp_insights['va_low']:,.1f}"],
                    "Distance": [f"{current_price - vp_insights['poc']:+.1f}", 
                               f"{current_price - vp_insights['va_high']:+.1f}", 
                               f"{current_price - vp_insights['va_low']:+.1f}"],
                    "Type": ["Support/Resistance", "Resistance", "Support"]
                }
                st.dataframe(pd.DataFrame(levels_data), use_container_width=True)
                
                # Trading recommendations based on VP
                st.subheader("🎯 VP Trading Strategy")
                if vp_insights['bias'] == 'BULLISH' and vp_insights['strength'] in ['STRONG', 'MODERATE']:
                    st.success("💡 **Bullish Setup**: Price above POC with strong volume support. Look for CALL options on dips to VA Low.")
                elif vp_insights['bias'] == 'BEARISH' and vp_insights['strength'] in ['STRONG', 'MODERATE']:
                    st.error("💡 **Bearish Setup**: Price below POC with strong volume resistance. Look for PUT options on rallies to VA High.")
                elif "INSIDE VALUE AREA" in vp_insights['price_vs_va']:
                    st.info("💡 **Range Trading**: Price inside Value Area. Wait for breakout above VA High (bullish) or below VA Low (bearish).")
                else:
                    st.warning("💡 **Neutral**: No clear VP signal. Wait for price to interact with key VP levels.")
    
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time}")
    
    # Enhanced test telegram with VP info
    if st.sidebar.button("Test Telegram"):
        if not df.empty and len(df) > 50:
            profiles = calculate_volume_profile(df, VP_CONFIG["TIMEFRAME"], VP_CONFIG["BINS"])
            vp_insights = get_volume_profile_insights(profiles, current_price)
            
            test_message = f"""
🔔 TEST MESSAGE - Nifty Analyzer with Volume Profile

📍 Current Price: ₹{current_price:.2f}
💰 Volume Profile Analysis:
📊 POC: ₹{vp_insights.get('poc', 0):.1f}
📈 VA High: ₹{vp_insights.get('va_high', 0):.1f}
📉 VA Low: ₹{vp_insights.get('va_low', 0):.1f}
🎯 Position: {vp_insights.get('price_vs_poc', 'N/A')}
💪 Bias: {vp_insights.get('strength', 'WEAK')} {vp_insights.get('bias', 'NEUTRAL')}

🕐 {current_time}
"""
        else:
            test_message = "🔔 Test message from Enhanced Nifty Analyzer with Volume Profile"
        
        send_telegram(test_message)
        st.sidebar.success("Test sent!")
    
    # Add Volume Profile explanation
    with st.expander("📚 Volume Profile Explanation"):
        st.markdown("""
        ### Volume Profile Analysis
        
        **Point of Control (POC)**: Price level with highest traded volume - acts as strong support/resistance
        
        **Value Area (VA)**: Price range containing 70% of total volume
        - **VA High**: Upper boundary of value area (resistance)
        - **VA Low**: Lower boundary of value area (support)
        
        **Trading Signals**:
        - 🟢 **Bullish**: Price above POC with strong volume
        - 🔴 **Bearish**: Price below POC with strong volume
        - ⚪ **Neutral**: Price near POC or mixed signals
        
        **Strength Levels**:
        - **STRONG**: Price outside Value Area
        - **MODERATE**: Price at VA boundaries
        - **WEAK**: Price inside Value Area
        """)

if __name__ == "__main__":
    main()
