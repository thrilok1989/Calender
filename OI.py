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
st.set_page_config(page_title="Nifty Analyzer + Volume Profile", page_icon="📈", layout="wide")

# Function to check if it's market hours
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
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
        """
        Fetch intraday historical data (OHLC + volume) using /charts/intraday endpoint.
        Dhan API supports intervals: 1,5,15,25,60.  
        Only last ~90 days of data possible in a single call for large intervals.  
        Source: DhanHQ docs. :contentReference[oaicite:0]{index=0}
        """
        url = "https://api.dhan.co/v2/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=days_back)
        
        payload = {
            "securityId": str(NIFTY_SCRIP),
            "exchangeSegment": NIFTY_SEG,
            "instrument": "INDEX",   # since underlying is index
            "interval": str(interval),  # string type
            "oi": False,  # you can set True if you want OI
            "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Intraday data fetch error: {response.status_code} {response.text}")
                return None
        except Exception as e:
            st.error(f"Exception during intraday data fetch: {str(e)}")
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
    except Exception as e:
        st.error(f"Telegram send error: {str(e)}")

def process_candle_data_intraday(data):
    """
    Converts intraday API response to dataframe including volume and datetime.
    """
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
        max_vals = resampled['high'].rolling(window=length * 2 + 1, center=True).max()
        min_vals = resampled['low'].rolling(window=length * 2 + 1, center=True).min()

        pivots = []
        for timestamp, value in resampled['high'][resampled['high'] == max_vals].items():
            pivots.append({'type': 'high', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        for timestamp, value in resampled['low'][resampled['low'] == min_vals].items():
            pivots.append({'type': 'low', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        return pivots
    except Exception as e:
        st.warning(f"Pivot calc error: {str(e)}")
        return []

def compute_volume_profile(df_full, period="1D", BINS=20):
    """
    Compute POC (Point of Control) for periods: e.g. each day, or last N intervals.
    period: "1D", "1W", etc.
    Returns list of dicts with: time period, high, low, poc, bins, hist
    """
    if df_full.empty:
        return []
    # Resample
    df = df_full.set_index('datetime')
    try:
        ohlc = df.resample(period).agg({
            "high": "max",
            "low": "min"
        }).dropna()
    except Exception as e:
        st.warning(f"Resample error in VP: {str(e)}")
        return []
    
    profiles = []
    for period_time, row in ohlc.iterrows():
        high = row['high']
        low = row['low']
        if high <= low:
            continue
        step = (high - low) / BINS
        bins = np.linspace(low, high, BINS + 1)
        hist = np.zeros(BINS)
        
        # sub_df: within the period
        mask = (df_full['datetime'] >= period_time) & (df_full['datetime'] < period_time + pd.to_timedelta(period))
        sub = df_full.loc[mask]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            # find which bin the close belongs to
            # using np.digitize is faster
            # But ensuring if at top bin, fall in last
            bin_index = np.digitize(r['close'], bins) - 1
            if bin_index < 0:
                bin_index = 0
            if bin_index >= BINS:
                bin_index = BINS - 1
            hist[bin_index] += r['volume']
        poc_idx = int(np.argmax(hist))
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        profiles.append({
            "period_time": period_time,
            "high": high,
            "low": low,
            "poc": poc,
            "bins": bins,
            "hist": hist
        })
    return profiles

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

def check_signals(df, option_data, current_price, proximity=5, poc_levels=None):
    """
    poc_levels: list of recent POC values, maybe from volume profile
    You can use Poc to filter / confirm signals
    """
    if df.empty or option_data is None or current_price is None:
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
    
    # PRIMARY SIGNAL
    pivots = get_pivots(df, "5") + get_pivots(df, "10") + get_pivots(df, "15")
    near_pivot = False
    pivot_level = None
    
    for pivot in pivots:
        if abs(current_price - pivot['value']) <= proximity:
            near_pivot = True
            pivot_level = pivot
            break
    
    # Use POC levels to further filter if provided
    near_poc = False
    poc_value = None
    if poc_levels:
        # check the most recent POC (e.g. last period) closeness
        poc_value = poc_levels[-1]  # you may choose which POC to use
        if abs(current_price - poc_value) <= proximity:
            near_poc = True
    
    # Only send primary if near pivot AND near POC (if using POC filter)
    if near_pivot and (not poc_levels or near_poc):
        primary_bullish_signal = (row['Level'] == 'Support' and bias_aligned_bullish)
        primary_bearish_signal = (row['Level'] == 'Resistance' and bias_aligned_bearish)
        
        if primary_bullish_signal or primary_bearish_signal:
            signal_type = "CALL" if primary_bullish_signal else "PUT"
            price_diff = current_price - pivot_level['value']
            
            message = f"""
🚨 PRIMARY NIFTY {signal_type} SIGNAL 🚨

📍 Spot: ₹{current_price:.2f} ({'ABOVE' if price_diff > 0 else 'BELOW'} pivot by {price_diff:+.2f})
📌 Pivot: {pivot_level['timeframe']}M at ₹{pivot_level['value']:.2f}
🎯 ATM Strike: {row['Strike']}

Conditions: {row['Level']}, All Bias Aligned
ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Ask: {row['Ask_Bias']}, Bid: {row['Bid_Bias']}
POC: {poc_value:.2f} 

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
            send_telegram(message)
            st.success(f"🔔 PRIMARY {signal_type} signal sent!")
    
    # Secondary, Third, etc as before — you can also include poc filtering similarly

def create_chart_with_poc(df, profiles, title):
    """
    Draw candlestick chart + volume bar + POC horizontal lines
    profiles: list of VP profile dicts
    """
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    # Volume bar chart
    volume_colors = ['#00ff88' if c >= o else '#ff4444' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'], name='Volume',
        marker_color=volume_colors, opacity=0.7
    ), row=2, col=1)

    # Add POC lines
    for prof in profiles:
        poc = prof['poc']
        period_time = prof['period_time']
        # draw horizontal line for the duration
        fig.add_shape(type="line", x0=period_time, x1=period_time + pd.to_timedelta("1D"), 
                      y0=poc, y1=poc, line=dict(color="blue", width=1.5, dash="dash"), row=1, col=1)

    fig.update_layout(title=title, template='plotly_dark', height=600,
                      xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def main():
    st.title("📈 Nifty Analyzer + Volume Profile")

    # Settings in sidebar
    st.sidebar.header("Settings")
    interval = st.sidebar.selectbox("Timeframe (min)", ["1", "3", "5", "15", "25", "60"], index=2)
    proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
    enable_signals = st.sidebar.checkbox("Enable Signals", value=True)
    enable_vp = st.sidebar.checkbox("Enable Volume Profile (POC)", value=True)
    vp_period = st.sidebar.selectbox("VP Period", ["1D", "1W"], index=0)
    vp_bins = st.sidebar.slider("VP Bins", 5, 50, 20)

    api = DhanAPI()

    col1, col2 = st.columns([2,1])

    with col1:
        st.header("Chart + Volume + VP")

        intraday_data = api.get_intraday_data(interval=interval, days_back=1)
        df = process_candle_data_intraday(intraday_data) if intraday_data else pd.DataFrame()

        ltp_data = api.get_ltp_data()
        current_price = None
        if ltp_data and 'data' in ltp_data:
            for exchange, data in ltp_data['data'].items():
                for security_id, price_data in data.items():
                    current_price = price_data.get('last_price', 0)
                    break

        if current_price is None and not df.empty:
            current_price = df['close'].iloc[-1]

        if not df.empty:
            prev_close = df['close'].iloc[-2] if len(df) > 1 else df['close'].iloc[-1]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

            st.metric("Price", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            st.metric("High", f"₹{df['high'].max():,.2f}")
            st.metric("Low", f"₹{df['low'].min():,.2f}")
        else:
            st.warning("No intraday data for chart")

        profiles = []
        if enable_vp and not df.empty:
            profiles = compute_volume_profile(df, period=vp_period, BINS=vp_bins)

        if not df.empty:
            fig = create_chart_with_poc(df, profiles, f"Nifty {interval}min + VP")
            st.plotly_chart(fig, use_container_width=True)
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
                    # collect POC levels to use in signal check
                    poc_levels = [prof['poc'] for prof in profiles] if profiles else None
                    check_signals(df, option_summary, underlying_price, proximity, poc_levels)
            else:
                st.error("Options data unavailable")
        else:
            st.error("Expiry data unavailable")

    # Sidebar last update time
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time}")

    if st.sidebar.button("Test Telegram"):
        send_telegram("🔔 Test message from Nifty Analyzer + VP")
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()
