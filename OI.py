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

# ----------------------------
# Config / Constants
# ----------------------------
st.set_page_config(page_title="Nifty Analyzer + VP & Pivots", page_icon="📈", layout="wide")

# Credentials from secrets
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))

NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"

# Utility: check if market is open
def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5:  # weekend
        return False
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return market_start <= now <= market_end

if is_market_hours():
    st_autorefresh(interval=35000, key="autorefresh")
else:
    st.info("Market is closed. Auto-refresh disabled.")

# ----------------------------
# DhanAPI helper class
# ----------------------------
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
        Fetch intraday OHLC + volume via /charts/intraday
        interval in ["1","5","15","25","60"]
        """
        url = "https://api.dhan.co/v2/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_dt = datetime.now(ist)
        start_dt = end_dt - timedelta(days=days_back)
        payload = {
            "securityId": str(NIFTY_SCRIP),
            "exchangeSegment": NIFTY_SEG,
            "instrument": "INDEX",   
            "interval": str(interval),
            "oi": False,
            "fromDate": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": end_dt.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            resp = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Error fetching intraday data: {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            st.error(f"Exception in get_intraday_data: {e}")
            return None

    def get_ltp_data(self):
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = { NIFTY_SEG: [NIFTY_SCRIP] }
        try:
            resp = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"Error fetching LTP data: {resp.status_code} {resp.text}")
                return None
        except Exception as e:
            st.error(f"Exception in get_ltp_data: {e}")
            return None

# Options endpoints
def get_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {
        "UnderlyingScrip": NIFTY_SCRIP,
        "UnderlyingSeg": NIFTY_SEG,
        "Expiry": expiry
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Error fetching option chain: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        st.error(f"Exception in get_option_chain: {e}")
        return None

def get_expiry_list():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {
        "UnderlyingScrip": NIFTY_SCRIP,
        "UnderlyingSeg": NIFTY_SEG
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Error fetching expiry list: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        st.error(f"Exception in get_expiry_list: {e}")
        return None

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        _ = requests.post(url, json=payload, timeout=10)
    except Exception as e:
        st.error(f"Telegram send error: {e}")

# ----------------------------
# Data processing / indicators
# ----------------------------
def process_candle_data_intraday(data):
    if not data:
        return pd.DataFrame()
    required = ['timestamp','open','high','low','close','volume']
    for fld in required:
        if fld not in data:
            st.error(f"Field {fld} missing in intraday response")
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
    rule_map = {"1": "1min", "3": "3min", "5": "5min", "10": "10min", "15": "15min", "25": "25min", "60": "60min"}
    rule = rule_map.get(timeframe, "5min")
    df2 = df.set_index('datetime')
    try:
        resampled = df2.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
    except Exception as e:
        st.error(f"Resample error: {e}")
        return []
    # debug
    st.write(f"Resampled ({rule}) count: {len(resampled)}")
    if len(resampled) < (length*2 + 1):
        st.write("Not enough data to calculate pivots for this timeframe.")
        return []
    max_vals = resampled['high'].rolling(window=length*2 + 1, center=True).max()
    min_vals = resampled['low'].rolling(window=length*2 + 1, center=True).min()
    pivots = []
    for ts, high_val in resampled['high'][resampled['high'] == max_vals].items():
        pivots.append({'type': 'high', 'timeframe': timeframe, 'timestamp': ts, 'value': high_val})
    for ts, low_val in resampled['low'][resampled['low'] == min_vals].items():
        pivots.append({'type': 'low', 'timeframe': timeframe, 'timestamp': ts, 'value': low_val})
    return pivots

def compute_volume_profile(df_full, period="1D", bins=20):
    if df_full.empty:
        return []
    df_full2 = df_full.copy()
    df_full2 = df_full2.set_index('datetime')
    try:
        ohlc = df_full2.resample(period).agg({"high":"max","low":"min"}).dropna()
    except Exception as e:
        st.error(f"VP resample error: {e}")
        return []
    profiles = []
    for period_time, row in ohlc.iterrows():
        high = row['high']
        low = row['low']
        if high <= low:
            continue
        bin_edges = np.linspace(low, high, bins + 1)
        hist = np.zeros(bins)
        mask = (df_full['datetime'] >= period_time) & (df_full['datetime'] < period_time + pd.to_timedelta(period))
        sub = df_full.loc[mask]
        if sub.empty:
            continue
        # assign volumes
        # faster via numpy digitize
        closes = sub['close'].values
        volumes = sub['volume'].values
        inds = np.digitize(closes, bin_edges) - 1
        inds[inds < 0] = 0
        inds[inds >= bins] = bins - 1
        for i, v in zip(inds, volumes):
            hist[i] += v
        poc_idx = int(np.argmax(hist))
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
        profiles.append({
            "period_time": period_time,
            "high": high,
            "low": low,
            "poc": poc,
            "bins": bin_edges,
            "hist": hist
        })
    return profiles

def analyze_options(expiry):
    oc = get_option_chain(expiry)
    if not oc or 'data' not in oc:
        return None, None
    data = oc['data']
    underlying = data.get('last_price', None)
    oc_data = data.get('oc', {})
    
    calls = []
    puts = []
    for strike, sd in oc_data.items():
        try:
            if 'ce' in sd:
                ce = sd['ce']
                ce['strikePrice'] = float(strike)
                calls.append(ce)
            if 'pe' in sd:
                pe = sd['pe']
                pe['strikePrice'] = float(strike)
                puts.append(pe)
        except:
            pass
    if not calls or not puts:
        return underlying, None
    df_ce = pd.DataFrame(calls)
    df_pe = pd.DataFrame(puts)
    df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE','_PE')).sort_values('strikePrice')
    # rename columns
    rename_map = {
        'last_price':'lastPrice',
        'oi':'openInterest',
        'previous_oi':'previousOpenInterest',
        'top_ask_quantity':'askQty',
        'top_bid_quantity':'bidQty',
        'volume':'totalTradedVolume'
    }
    for old, new in rename_map.items():
        df.rename(columns={f"{old}_CE":f"{new}_CE", f"{old}_PE":f"{new}_PE"}, inplace=True)
    df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
    df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
    atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
    df_filtered = df[abs(df['strikePrice'] - atm_strike) <= 100]
    df_filtered['Zone'] = df_filtered['strikePrice'].apply(
        lambda x: 'ATM' if x == atm_strike else ('ITM' if x < underlying else 'OTM')
    )
    bias_results = []
    for _, r in df_filtered.iterrows():
        chg_oi_bias = "Bullish" if r['changeinOpenInterest_CE'] < r['changeinOpenInterest_PE'] else "Bearish"
        volume_bias = "Bullish" if r['totalTradedVolume_CE'] < r['totalTradedVolume_PE'] else "Bearish"
        ask_ce = r.get('askQty_CE', 0)
        ask_pe = r.get('askQty_PE', 0)
        bid_ce = r.get('bidQty_CE', 0)
        bid_pe = r.get('bidQty_PE', 0)
        ask_bias = "Bearish" if ask_ce > ask_pe else "Bullish"
        bid_bias = "Bullish" if bid_ce > bid_pe else "Bearish"
        ce_oi = r['openInterest_CE']
        pe_oi = r['openInterest_PE']
        level = "Support" if pe_oi > 1.12 * ce_oi else ("Resistance" if ce_oi > 1.12 * pe_oi else "Neutral")
        bias_results.append({
            "Strike": r['strikePrice'],
            "Zone": r['Zone'],
            "Level": level,
            "ChgOI_Bias": chg_oi_bias,
            "Volume_Bias": volume_bias,
            "Ask_Bias": ask_bias,
            "Bid_Bias": bid_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else float('inf'), 2),
            "changeinOpenInterest_CE": r['changeinOpenInterest_CE'],
            "changeinOpenInterest_PE": r['changeinOpenInterest_PE']
        })
    return underlying, pd.DataFrame(bias_results)

def check_signals(df, option_summary, current_price, proximity=5, pivots=None):
    if df.empty or option_summary is None or current_price is None:
        return
    atm = option_summary[option_summary['Zone'] == 'ATM']
    if atm.empty:
        return
    row = atm.iloc[0]
    ce_chg = abs(row.get('changeinOpenInterest_CE', 0))
    pe_chg = abs(row.get('changeinOpenInterest_PE', 0))
    bias_aligned_bullish = (row['ChgOI_Bias'] == 'Bullish' and 
                            row['Volume_Bias'] == 'Bullish' and 
                            row['Ask_Bias'] == 'Bullish' and 
                            row['Bid_Bias'] == 'Bullish')
    bias_aligned_bearish = (row['ChgOI_Bias'] == 'Bearish' and 
                            row['Volume_Bias'] == 'Bearish' and 
                            row['Ask_Bias'] == 'Bearish' and 
                            row['Bid_Bias'] == 'Bearish')
    # find pivot near price
    near_pivot = False
    pivot_level = None
    if pivots:
        for p in pivots:
            if abs(current_price - p['value']) <= proximity:
                near_pivot = True
                pivot_level = p
                break
    # primary signal
    if near_pivot:
        if row['Level'] == 'Support' and bias_aligned_bullish:
            send_telegram(f"✅ PRIMARY CALL signal: Spot {current_price:.2f}, Pivot {pivot_level['value']:.2f}")
            st.success("PRIMARY CALL signal sent")
        elif row['Level'] == 'Resistance' and bias_aligned_bearish:
            send_telegram(f"⚠️ PRIMARY PUT signal: Spot {current_price:.2f}, Pivot {pivot_level['value']:.2f}")
            st.success("PRIMARY PUT signal sent")

# ----------------------------
# Chart function with volume, pivots, and optional VP lines
# ----------------------------
def create_chart(df, vp_profiles, pivots, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
            name="Nifty"
        ),
        row=1, col=1
    )
    # Volume
    fig.add_trace(
        go.Bar(
            x=df['datetime'], y=df['volume'],
            marker_color=['#00ff88' if cls >= opn else '#ff4444' for cls, opn in zip(df['close'], df['open'])],
            name="Volume",
            opacity=0.6
        ),
        row=2, col=1
    )
    # POC / VP lines
    for prof in vp_profiles:
        poc = prof['poc']
        period_time = prof['period_time']
        # draw horizontal line from that period_time to end of df
        end_time = df['datetime'].max()
        fig.add_shape(
            type="line",
            x0=period_time, x1=end_time,
            y0=poc, y1=poc,
            line=dict(color="blue", width=1.5, dash="dash"),
            row=1, col=1
        )
        fig.add_annotation(
            x=period_time, y=poc,
            text=f"POC {poc:.2f}", showarrow=False,
            font=dict(size=8, color="white"),
            row=1, col=1
        )
    # Pivot lines
    for p in pivots:
        pv = p['value']
        ts = p['timestamp']
        end_time = df['datetime'].max()
        fig.add_shape(
            type="line",
            x0=ts, x1=end_time,
            y0=pv, y1=pv,
            line=dict(color="yellow", width=1, dash="dot"),
            row=1, col=1
        )
    fig.update_layout(title=title, template='plotly_dark', xaxis_rangeslider_visible=False, showlegend=False, height=600)
    return fig

# ----------------------------
# Main app
# ----------------------------
def main():
    st.title("📈 Nifty Analyzer with Volume Profile & Pivots")

    # Sidebar settings
    st.sidebar.header("Settings")
    interval = st.sidebar.selectbox("Interval (min)", ["1","3","5","15","25","60"], index=2)
    proximity = st.sidebar.slider("Proximity for signals/pivots", min_value=1, max_value=20, value=5)
    enable_vp = st.sidebar.checkbox("Enable Volume Profile (POC Lines)", value=True)
    vp_period = st.sidebar.selectbox("VP period", ["1D","1W"], index=0)
    vp_bins = st.sidebar.slider("VP bins", min_value=5, max_value=50, value=20)
    enable_signals = st.sidebar.checkbox("Enable Signals", value=True)

    api = DhanAPI()

    # Fetch data
    intr = api.get_intraday_data(interval=interval, days_back=1)
    df = process_candle_data_intraday(intr) if intr else pd.DataFrame()
    if df.empty:
        st.error("No intraday candle + volume data available")
        return

    # Show a sample to verify volume data
    st.write("Sample data (last 5 rows):", df[['datetime','open','close','volume']].tail(5))

    ltp_json = api.get_ltp_data()
    current_price = None
    if ltp_json and 'data' in ltp_json:
        for exch, dd in ltp_json['data'].items():
            for sid, pdict in dd.items():
                current_price = pdict.get('last_price', None)
                break
            break

    if current_price is None:
        current_price = df['close'].iloc[-1]

    # Compute pivots
    piv5 = get_pivots(df, timeframe="5", length=4)
    piv10 = get_pivots(df, timeframe="10", length=4)
    all_pivots = piv5 + piv10

    # Compute VP profiles if enabled
    vp_profiles = []
    if enable_vp:
        vp_profiles = compute_volume_profile(df_full=df, period=vp_period, bins=vp_bins)

    # Display chart
    fig = create_chart(df, vp_profiles, all_pivots, title=f"Nifty {interval}min Chart")
    st.plotly_chart(fig, use_container_width=True)

    # Show metrics
    st.metric("Current Price", f"₹{current_price:.2f}")

    # Options data and signals
    expiry_data = get_expiry_list()
    if expiry_data and 'data' in expiry_data:
        selected_exp = st.selectbox("Select Expiry", expiry_data['data'])
        underlying_price, opt_summary = analyze_options(selected_exp)
        if underlying_price and opt_summary is not None:
            st.dataframe(opt_summary, use_container_width=True)
            if enable_signals:
                check_signals(df, opt_summary, current_price, proximity, pivots=all_pivots)
        else:
            st.warning("Option data unavailable for this expiry")
    else:
        st.warning("Expiry list unavailable")

    # Update time
    ist = pytz.timezone('Asia/Kolkata')
    st.sidebar.info(f"Last update: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')}")

    # Test Telegram
    if st.sidebar.button("Test Telegram"):
        send_telegram("🔔 Test message from Nifty Analyzer")
        st.sidebar.success("Test sent")

if __name__ == "__main__":
    main()
