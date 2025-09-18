import streamlit as st from streamlit_autorefresh import st_autorefresh import requests import pandas as pd import plotly.graph_objects as go from plotly.subplots import make_subplots import datetime import pytz import numpy as np import math from scipy.stats import norm from datetime import datetime, timedelta

Page config

st.set_page_config(page_title="Nifty Analyzer", page_icon="📈", layout="wide")

---------- CONFIG ----------

CONTRACT_SIZE = 15  # Nifty lot size (adjust if needed) RISK_FREE_RATE = 0.07  # annual risk-free rate for BS (approx). Adjust if you want.

Function to check if it's market hours

def is_market_hours(): ist = pytz.timezone('Asia/Kolkata') now = datetime.now(ist) if now.weekday() >= 5: return False market_start = now.replace(hour=9, minute=0, second=0, microsecond=0) market_end = now.replace(hour=15, minute=45, second=0, microsecond=0) return market_start <= now <= market_end

Only run autorefresh during market hours

if is_market_hours(): st_autorefresh(interval=35000, key="refresh") else: st.info("Market is closed. Auto-refresh disabled.")

Credentials

DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "") DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "") TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "") TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", "")) NIFTY_SCRIP = 13 NIFTY_SEG = "IDX_I"

class DhanAPI: def init(self): self.headers = { 'Accept': 'application/json', 'Content-Type': 'application/json', 'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID } def get_intraday_data(self, interval="5", days_back=1): url = "https://api.dhan.co/v2/charts/intraday" ist = pytz.timezone('Asia/Kolkata') end_date = datetime.now(ist) start_date = end_date - timedelta(days=days_back) payload = { "securityId": str(NIFTY_SCRIP), "exchangeSegment": NIFTY_SEG, "instrument": "INDEX", "interval": interval, "oi": False, "fromDate": start_date.strftime("%Y-%m-%d %H:%M:%S"), "toDate": end_date.strftime("%Y-%m-%d %H:%M:%S") } try: response = requests.post(url, headers=self.headers, json=payload, timeout=10) return response.json() if response.status_code == 200 else None except: return None def get_ltp_data(self): url = "https://api.dhan.co/v2/marketfeed/ltp" payload = {NIFTY_SEG: [NIFTY_SCRIP]} try: response = requests.post(url, headers=self.headers, json=payload, timeout=10) return response.json() if response.status_code == 200 else None except: return None

Option chain helpers

def get_option_chain(expiry): url = "https://api.dhan.co/v2/optionchain" headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'} payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry} try: response = requests.post(url, headers=headers, json=payload, timeout=10) return response.json() if response.status_code == 200 else None except: return None

def get_expiry_list(): url = "https://api.dhan.co/v2/optionchain/expirylist" headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'} payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG} try: response = requests.post(url, headers=headers, json=payload, timeout=10) return response.json() if response.status_code == 200 else None except: return None

def send_telegram(message): if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"} try: requests.post(url, json=payload, timeout=10) except: pass

Candle processing

def process_candle_data(data): if not data or 'open' not in data: return pd.DataFrame() df = pd.DataFrame({ 'timestamp': data['timestamp'], 'open': data['open'], 'high': data['high'], 'low': data['low'], 'close': data['close'], 'volume': data['volume'] }) ist = pytz.timezone('Asia/Kolkata') df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist) return df

Indicators

def calculate_rsi(data, period=14): delta = data['close'].diff() gain = (delta.where(delta > 0, 0)).rolling(window=period).mean() loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean() rs = gain / loss rsi = 100 - (100 / (1 + rs)) return rsi

def calculate_vwap(df): # VWAP on underlying using candle typical price * volume if df.empty: return pd.Series() tp = (df['high'] + df['low'] + df['close']) / 3 cum_turnover = (tp * df['volume']).cumsum() cum_volume = df['volume'].cumsum() vwap = (cum_turnover / cum_volume).fillna(method='ffill') return vwap

def calculate_atr(df, period=14): if df.empty: return pd.Series() high_low = df['high'] - df['low'] high_close = (df['high'] - df['close'].shift()).abs() low_close = (df['low'] - df['close'].shift()).abs() tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1) atr = tr.rolling(window=period).mean() return atr

Black-Scholes (European) price for option - used for IV

def bs_price(S, K, T, r, sigma, option_type='call'): if T <= 0: if option_type == 'call': return max(0.0, S - K) else: return max(0.0, K - S) d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T)) d2 = d1 - sigma * math.sqrt(T) if option_type == 'call': return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2) else: return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

Implied volatility via bisection

def implied_volatility(market_price, S, K, T, r, option_type='call', tol=1e-4, max_iter=100): if market_price <= 0: return 0.0 low, high = 1e-6, 5.0 for i in range(max_iter): mid = (low + high) / 2 p = bs_price(S, K, T, r, mid, option_type) if abs(p - market_price) < tol: return mid if p > market_price: high = mid else: low = mid return mid

Option gamma (BS formula)

def option_gamma(S, K, T, r, sigma): if T <= 0 or sigma <= 0: return 0.0 d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T)) return norm.pdf(d1) / (S * sigma * math.sqrt(T))

Calculate IV, GEX and Volume Spread (approx) from option chain

def enrich_option_chain(raw_data, underlying_price, expiry_date): # raw_data: oc dict from DHAN calls, puts = [], [] for strike, strike_data in raw_data.items(): strike_f = float(strike) if 'ce' in strike_data: ce = strike_data['ce'].copy() ce['strikePrice'] = strike_f ce['type'] = 'CE' calls.append(ce) if 'pe' in strike_data: pe = strike_data['pe'].copy() pe['strikePrice'] = strike_f pe['type'] = 'PE' puts.append(pe) df_ce = pd.DataFrame(calls) df_pe = pd.DataFrame(puts) if df_ce.empty and df_pe.empty: return pd.DataFrame() # merge safely on strike df = pd.merge(df_ce, df_pe, on='strikePrice', how='outer', suffixes=('_CE', '_PE'))

# time to expiry in years (approx)
try:
    exp_dt = pd.to_datetime(expiry_date)
except:
    exp_dt = datetime.now(pytz.timezone('Asia/Kolkata'))
now = datetime.now(pytz.timezone('Asia/Kolkata'))
T = max((exp_dt - now).total_seconds(), 0) / (365 * 24 * 3600)

iv_list = []
gex_list = []
delta_buy = []
delta_sell = []

for _, row in df.iterrows():
    # CE
    try:
        ce_ltp = float(row.get('last_price_CE', 0) or 0)
        ce_bid = float(row.get('top_bid_price_CE', 0) or 0)
        ce_ask = float(row.get('top_ask_price_CE', 0) or 0)
        ce_oi = float(row.get('oi_CE', 0) or 0)
        ce_vol = float(row.get('totalTradedVolume_CE', 0) or 0)
    except:
        ce_ltp = ce_bid = ce_ask = ce_oi = ce_vol = 0
    # PE
    try:
        pe_ltp = float(row.get('last_price_PE', 0) or 0)
        pe_bid = float(row.get('top_bid_price_PE', 0) or 0)
        pe_ask = float(row.get('top_ask_price_PE', 0) or 0)
        pe_oi = float(row.get('oi_PE', 0) or 0)
        pe_vol = float(row.get('totalTradedVolume_PE', 0) or 0)
    except:
        pe_ltp = pe_bid = pe_ask = pe_oi = pe_vol = 0

    K = float(row['strikePrice'])
    # IV (approx) - use last traded price classification
    ce_iv = implied_volatility(ce_ltp, underlying_price, K, T, RISK_FREE_RATE, 'call') if ce_ltp > 0 else 0
    pe_iv = implied_volatility(pe_ltp, underlying_price, K, T, RISK_FREE_RATE, 'put') if pe_ltp > 0 else 0

    # gamma per option
    ce_gamma = option_gamma(underlying_price, K, T, RISK_FREE_RATE, ce_iv) if ce_iv > 0 else 0
    pe_gamma = option_gamma(underlying_price, K, T, RISK_FREE_RATE, pe_iv) if pe_iv > 0 else 0

    # approximate GEX contribution: gamma * OI * contract_size * underlying^2 (relative scale)
    ce_gex = ce_gamma * ce_oi * CONTRACT_SIZE * (underlying_price ** 2)
    pe_gex = pe_gamma * pe_oi * CONTRACT_SIZE * (underlying_price ** 2)

    # Volume spread approx: classify last traded volume as buy or sell depending on LTP vs bid/ask
    # Note: this is an approximation unless tick data is available
    ce_buy = ce_sell = 0
    if ce_ltp and ce_ask and ce_bid:
        if abs(ce_ltp - ce_ask) < 1e-9 or ce_ltp >= ce_ask:
            ce_buy = ce_vol
        elif abs(ce_ltp - ce_bid) < 1e-9 or ce_ltp <= ce_bid:
            ce_sell = ce_vol
        else:
            # ambiguous trade between bid and ask -> split
            ce_buy = ce_vol / 2
            ce_sell = ce_vol / 2
    pe_buy = pe_sell = 0
    if pe_ltp and pe_ask and pe_bid:
        if abs(pe_ltp - pe_ask) < 1e-9 or pe_ltp >= pe_ask:
            pe_buy = pe_vol
        elif abs(pe_ltp - pe_bid) < 1e-9 or pe_ltp <= pe_bid:
            pe_sell = pe_vol
        else:
            pe_buy = pe_vol / 2
            pe_sell = pe_vol / 2

    iv_list.append({'strike': K, 'ce_iv': ce_iv, 'pe_iv': pe_iv})
    gex_list.append({'strike': K, 'ce_gex': ce_gex, 'pe_gex': pe_gex})
    delta_buy.append({'strike': K, 'ce_buy': ce_buy, 'pe_buy': pe_buy})
    delta_sell.append({'strike': K, 'ce_sell': ce_sell, 'pe_sell': pe_sell})

iv_df = pd.DataFrame(iv_list)
gex_df = pd.DataFrame(gex_list)
buy_df = pd.DataFrame(delta_buy)
sell_df = pd.DataFrame(delta_sell)

merged = df.merge(iv_df, left_on='strikePrice', right_on='strike', how='left')
merged = merged.merge(gex_df, left_on='strikePrice', right_on='strike', how='left', suffixes=('', '_gex'))
merged = merged.merge(buy_df, left_on='strikePrice', right_on='strike', how='left')
merged = merged.merge(sell_df, left_on='strikePrice', right_on='strike', how='left', suffixes=('', '_sell'))

# fillna
merged.fillna(0, inplace=True)

# compute deltas and cumulative across strikes
merged['ce_delta'] = merged['ce_buy'] - merged['ce_sell']
merged['pe_delta'] = merged['pe_buy'] - merged['pe_sell']

# net GEX across chain (positive means net long gamma from market perspective)
merged['net_gex'] = merged['ce_gex'] - merged['pe_gex']

return merged

Pivot code (unchanged)

def get_pivots(df, timeframe="5", length=4): if df.empty: return [] rule_map = {"3": "3min", "5": "5min", "10": "10min", "15": "15min"} rule = rule_map.get(timeframe, "5min") df_temp = df.set_index('datetime') try: resampled = df_temp.resample(rule).agg({ "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum" }).dropna() if len(resampled) < length * 2 + 1: return [] max_vals = resampled['high'].rolling(window=length2+1, center=True).max() min_vals = resampled['low'].rolling(window=length2+1, center=True).min() pivots = [] for timestamp, value in resampled['high'][resampled['high'] == max_vals].items(): pivots.append({'type': 'high', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value}) for timestamp, value in resampled['low'][resampled['low'] == min_vals].items(): pivots.append({'type': 'low', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value}) return pivots except: return []

Chart creation with VWAP band plotting

def create_chart(df, title): if df.empty: return go.Figure() df['rsi'] = calculate_rsi(df) df['vwap'] = calculate_vwap(df) df['atr'] = calculate_atr(df) fig = make_subplots( rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], subplot_titles=('Price', 'Volume', 'RSI') ) fig.add_trace(go.Candlestick( x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Nifty', increasing_line_color='#00ff88', decreasing_line_color='#ff4444' ), row=1, col=1) # VWAP line fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap'], name='VWAP', line=dict(width=2)), row=1, col=1) # VWAP bands (1sigma and 2sigma) computed on typical price deviations tp = (df['high'] + df['low'] + df['close']) / 3 rolling_std = tp.rolling(window=20).std() fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap'] + rolling_std, name='VWAP+1σ', line=dict(width=1, dash='dash')), row=1, col=1) fig.add_trace(go.Scatter(x=df['datetime'], y=df['vwap'] - rolling_std, name='VWAP-1σ', line=dict(width=1, dash='dash')), row=1, col=1) # Volume volume_colors = ['#00ff88' if c >= o else '#ff4444' for c,o in zip(df['close'], df['open'])] fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], name='Volume', marker_color=volume_colors, opacity=0.7), row=2, col=1) # RSI fig.add_trace(go.Scatter(x=df['datetime'], y=df['rsi'], name='RSI', line=dict(color='#ff9900', width=2)), row=3, col=1) fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1) fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1) fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1) if len(df) > 50: timeframes = ["5", "10", "15"] colors = ["#ff9900", "#ff44ff", '#4444ff'] for tf, color in zip(timeframes, colors): pivots = get_pivots(df, tf) x_start, x_end = df['datetime'].min(), df['datetime'].max() for pivot in pivots[-5:]: fig.add_shape(type="line", x0=x_start, x1=x_end, y0=pivot['value'], y1=pivot['value'], line=dict(color=color, width=1, dash="dash"), row=1, col=1) fig.update_layout(title=title, template='plotly_dark', height=900, xaxis_rangeslider_visible=False, showlegend=False) fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1) return fig

Analyze options with enrichment

def analyze_options(expiry): option_data = get_option_chain(expiry) if not option_data or 'data' not in option_data: return None, None data = option_data['data'] underlying = data['last_price'] oc_data = data['oc'] enriched = enrich_option_chain(oc_data, underlying, expiry) if enriched.empty: return underlying, None # Find ATM and filter ±2 strikes atm_strike = min(enriched['strikePrice'], key=lambda x: abs(x - underlying)) strikes_range = enriched[(abs(enriched['strikePrice'] - atm_strike) <= 200)]  # approx ±2 strikes (adjust step) strikes_range['Zone'] = strikes_range['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')

# Aggregate CE vs PE delta and GEX on the filtered range
total_ce_delta = strikes_range['ce_delta'].sum()
total_pe_delta = strikes_range['pe_delta'].sum()
total_ce_gex = strikes_range['ce_gex'].sum()
total_pe_gex = strikes_range['pe_gex'].sum()

# cumulative delta series (for display)
strikes_sorted = strikes_range.sort_values('strikePrice')
strikes_sorted['cum_ce_delta'] = strikes_sorted['ce_delta'].cumsum()
strikes_sorted['cum_pe_delta'] = strikes_sorted['pe_delta'].cumsum()

# Add IV avg
strikes_sorted['iv_mid'] = (strikes_sorted['ce_iv'] + strikes_sorted['pe_iv']) / 2

# Build summary
summary = strikes_sorted[['strikePrice', 'Zone', 'ce_iv', 'pe_iv', 'iv_mid', 'ce_delta', 'pe_delta', 'cum_ce_delta', 'cum_pe_delta', 'ce_gex', 'pe_gex']].copy()
summary.rename(columns={'strikePrice': 'Strike', 'ce_iv':'IV_CE', 'pe_iv':'IV_PE', 'iv_mid':'IV_Mid', 'ce_delta':'CE_Delta','pe_delta':'PE_Delta','cum_ce_delta':'CE_CumDelta','cum_pe_delta':'PE_CumDelta','ce_gex':'CE_GEX','pe_gex':'PE_GEX'}, inplace=True)

# Bias score (simple weighted combination) - tweak weights as needed
# Weights: CE vs PE delta 30%, IV skew 20%, GEX net 25%, PCR/OI bias 15%, VWAP position 10% (VWAP will be applied later at main())
summary['delta_score'] = (summary['CE_CumDelta'] - summary['PE_CumDelta'])
summary['gex_score'] = (summary['CE_GEX'] - summary['PE_GEX'])
summary['iv_skew'] = (summary['IV_CE'] - summary['IV_PE'])

# Normalize scores to comparable ranges
def zscore(s):
    if s.std() == 0:
        return s * 0
    return (s - s.mean()) / (s.std() + 1e-9)
summary['delta_z'] = zscore(summary['delta_score'])
summary['gex_z'] = zscore(summary['gex_score'])
summary['iv_z'] = zscore(summary['iv_skew'])

summary['bias_score'] = 0.4 * summary['delta_z'] + 0.35 * summary['gex_z'] + 0.25 * summary['iv_z']
# positive bias_score -> bullish (calls dominate), negative -> bearish
summary['Bias'] = summary['bias_score'].apply(lambda x: 'Bullish' if x > 0.3 else ('Bearish' if x < -0.3 else 'Neutral'))

return underlying, summary

Signals - updated to use enriched data

def check_signals(df, option_data, current_price, proximity=5): if df.empty or option_data is None or not current_price: return df['rsi'] = calculate_rsi(df) current_rsi = df['rsi'].iloc[-1] if not df.empty else None

atm_data = option_data[option_data['Zone'] == 'ATM'] if 'Zone' in option_data.columns else option_data[option_data['Strike'] == option_data['Strike'].iloc[0]]
if atm_data.empty:
    return
row = atm_data.iloc[0]

# Use bias_score for signal confirmation
bias_label = row.get('Bias', 'Neutral')
ce_cum = row.get('CE_CumDelta', 0)
pe_cum = row.get('PE_CumDelta', 0)

# Pivot proximity
pivots = get_pivots(df, "5") + get_pivots(df, "10") + get_pivots(df, "15")
near_pivot = False
pivot_level = None
for pivot in pivots:
    if abs(current_price - pivot['value']) <= proximity:
        near_pivot = True
        pivot_level = pivot
        break

# Primary signal based on Bias and pivot
if near_pivot and pivot_level is not None:
    if bias_label == 'Bullish' and row.get('IV_Mid', 0) < 1:  # IV_Mid small placeholder - keep for demonstration
        send_telegram(f"📈 PRIMARY CALL SIGNAL - Spot {current_price} | ATM {row['Strike']} | RSI {current_rsi:.2f}")
        st.success("🔔 PRIMARY CALL signal sent!")
    elif bias_label == 'Bearish' and row.get('IV_Mid', 0) < 1:
        send_telegram(f"📉 PRIMARY PUT SIGNAL - Spot {current_price} | ATM {row['Strike']} | RSI {current_rsi:.2f}")
        st.success("🔔 PRIMARY PUT signal sent!")

# RSI extremes (unchanged)
if current_rsi is not None:
    if current_rsi > 70:
        send_telegram(f"⚠️ RSI Overbought - Spot {current_price} | RSI {current_rsi:.2f}")
        st.success("⚠️ RSI Overbought signal sent!")
    elif current_rsi < 30:
        send_telegram(f"⚠️ RSI Oversold - Spot {current_price} | RSI {current_rsi:.2f}")
        st.success("⚠️ RSI Oversold signal sent!")

Main app

def main(): st.title("📈 Nifty Trading Analyzer - Upgraded") ist = pytz.timezone('Asia/Kolkata') current_time = datetime.now(ist) if not is_market_hours(): st.warning(f"⚠️ Market is closed. Current time: {current_time.strftime('%H:%M:%S IST')}") st.info("Market hours: Monday-Friday, 9:00 AM to 3:45 PM IST")

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
        for exchange, data2 in ltp_data['data'].items():
            for security_id, price_data in data2.items():
                current_price = price_data.get('last_price', 0)
                break
    if current_price is None and not df.empty:
        current_price = df['close'].iloc[-1]
    if not df.empty and len(df) > 1:
        prev_close = df['close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        current_rsi = None
        if not df.empty:
            df['rsi'] = calculate_rsi(df)
            current_rsi = df['rsi'].iloc[-1]
        col1_m, col2_m, col3_m, col4_m = st.columns(4)
        with col1_m:
            st.metric("Price", f"₹{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
        with col2_m:
            st.metric("High", f"₹{df['high'].max():,.2f}")
        with col3_m:
            st.metric("Low", f"₹{df['low'].min():,.2f}")
        with col4_m:
            st.metric("RSI", f"{current_rsi:.2f}" if current_rsi is not None else "N/A")
    if not df.empty:
        # compute VWAP & ATR for display
        df['vwap'] = calculate_vwap(df)
        df['atr'] = calculate_atr(df)
        fig = create_chart(df, f"Nifty {interval}min")
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

current_time = datetime.now(ist).strftime("%H:%M:%S IST")
st.sidebar.info(f"Updated: {current_time}")
if st.sidebar.button("Test Telegram"):
    send_telegram("🔔 Test message from Nifty Analyzer")
    st.sidebar.success("Test sent!")

if name == "main": main()

