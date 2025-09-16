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

# Page config
st.set_page_config(page_title="Nifty Analyzer", page_icon="📈", layout="wide")

# Check if market hours (Mon-Fri 9AM-3:45PM IST)
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return market_start <= now <= market_end

# Auto-refresh only during market hours
if is_market_hours():
    st_autorefresh(interval=80000, key="refresh")
else:
    st.info("Market is closed. Auto-refresh disabled.")

# Credentials from secrets
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

    def get_intraday_data(self, interval="1", days_back=1):  # default interval 1 minute
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

def create_chart(
    df, title, bins=10, volume_profile_timeframe="1D", enable_dynamic_poc=True):
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Nifty',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'),
        row=1, col=1)

    # Add pivot lines (if enough data)
    if len(df) > 50:
        timeframes = ["5", "10", "15"]
        colors = ["#ff9900", "#ff44ff", "#4444ff"]
        for tf, color in zip(timeframes, colors):
            pivots = get_pivots(df, tf)
            x_start, x_end = df['datetime'].min(), df['datetime'].max()
            for pivot in pivots[-5:]:
                fig.add_shape(type="line", x0=x_start, x1=x_end,
                              y0=pivot['value'], y1=pivot['value'],
                              line=dict(color=color, width=1, dash="dash"),
                              row=1, col=1)

    # Volume bars with color by candle direction
    if 'volume' in df and not df['volume'].isnull().all():
        volume_colors = ['#00ff88' if c >= o else '#ff4444'
                         for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df['datetime'], y=df['volume'], name='Volume',
            marker_color=volume_colors, opacity=0.7),
            row=2, col=1)
    else:
        st.warning("Volume data missing or empty; volume bars not shown.")

    # Resample for volume profile with dynamic timeframe & bins
    freq = volume_profile_timeframe.lower()
    try:
        ohlc = df.resample(freq, on='datetime').agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
    except Exception as e:
        st.error(f"Error in resampling: {e}")
        ohlc = pd.DataFrame()

    # Volume Profile & POC per resampled bar
    for idx, row_ohlc in ohlc.iterrows():
        high, low = row_ohlc['high'], row_ohlc['low']
        if high == low:
            bins_edges = np.linspace(low*0.99, high*1.01, bins + 1)
        else:
            bins_edges = np.linspace(low, high, bins + 1)
        step = (high - low) / bins if high > low else 1

        mask = (df['datetime'] >= idx) & (df['datetime'] < idx + pd.Timedelta(freq))
        sub_df = df.loc[mask]

        hist = np.zeros(bins)
        for _, r in sub_df.iterrows():
            for i in range(bins):
                if bins_edges[i] <= r['close'] < bins_edges[i + 1]:
                    hist[i] += r.get('volume', 0)
                    break
        if hist.max() == 0:
            continue
        hist_scaled = hist / hist.max() * 0.4 * (high - low)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        fig.add_trace(go.Bar(
            x=-hist_scaled,
            y=bin_centers,
            base=idx,
            orientation='h',
            width=step * 0.9,
            marker=dict(color='lightgray', line=dict(color='gray', width=0.5)),
            name=f"Vol Profile {idx.strftime('%Y-%m-%d %H:%M')}",
            showlegend=False,
            opacity=0.5
        ), row=1, col=1)

        if enable_dynamic_poc:
            poc_idx = hist.argmax()
            poc_price = bin_centers[poc_idx]
            fig.add_shape(type="line",
                          x0=idx, x1=idx + pd.Timedelta(freq),
                          y0=poc_price, y1=poc_price,
                          line=dict(color="blue", width=1, dash="dash"),
                          row=1, col=1)
            fig.add_annotation(
                x=idx + pd.Timedelta(freq) / 2,
                y=poc_price,
                text=f"POC\n{poc_price:.2f}",
                showarrow=False,
                yshift=10,
                font=dict(color="blue", size=10),
                row=1, col=1
            )

    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=False)
    return fig

# (Include your analyze_options, check_signals, and other functions unchanged here)

def main():
    st.title("📈 Nifty Trading Analyzer")
    
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    if not is_market_hours():
        st.warning(f"⚠️ Market is closed. Current time: {current_time.strftime('%H:%M:%S IST')}")
        st.info("Market hours: Monday-Friday, 9:00 AM to 3:45 PM IST")
    
    st.sidebar.header("Settings")
    interval = st.sidebar.selectbox("Chart Timeframe (minutes)", ["1", "3", "5", "10", "15"], index=0)  # default 1min
    proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
    enable_signals = st.sidebar.checkbox("Enable Signals", value=True)
    volume_profile_timeframe = st.sidebar.selectbox(
        "Volume Profile Timeframe", ["1D", "1W", "1M"], index=0)  # default 1 day
    volume_profile_bins = st.sidebar.slider("Volume Profile Bins", 5, 50, 10)  # default 10 bins
    enable_dynamic_poc = st.sidebar.checkbox("Enable Dynamic POC", value=True)  # default enabled
    
    api = DhanAPI()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Chart")

        data = api.get_intraday_data(interval)
        df = process_candle_data(data) if data else pd.DataFrame()

        ltp_data = api.get_ltp_data()
        current_price = None
        if ltp_data and 'data' in ltp_data:
            for exchange, data_ in ltp_data['data'].items():
                for security_id, price_data in data_.items():
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
            fig = create_chart(df, f"Nifty {interval}min",
                               bins=volume_profile_bins,
                               volume_profile_timeframe=volume_profile_timeframe,
                               enable_dynamic_poc=enable_dynamic_poc)
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
                    check_signals(df, option_summary, underlying_price, proximity)
            else:
                st.error("Options data unavailable")
        else:
            st.error("Expiry data unavailable")

    current_time_str = datetime.now(ist).strftime("%H:%M:%S IST")
    st.sidebar.info(f"Updated: {current_time_str}")

    if st.sidebar.button("Test Telegram"):
        send_telegram("🔔 Test message from Nifty Analyzer")
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()
