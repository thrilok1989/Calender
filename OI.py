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
import time

st.set_page_config(page_title="Nifty Analyzer", page_icon="📈", layout="wide")

# Credentials
DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(st.secrets.get("TELEGRAM_CHAT_ID", ""))

NIFTY_SCRIP = 13
NIFTY_SEG = "IDX_I"
NIFTY_FUTURES_SCRIP = 53001
NIFTY_FUTURES_SEG = "NSE_FNO"

VP_CONFIG = {"BINS": 20, "POC_COLOR": "#FFD700", "VOLUME_COLOR": "#1f77b4"}

def is_market_hours():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5: return False
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return market_start <= now <= market_end

if is_market_hours():
    st_autorefresh(interval=120000, key="refresh")

class DhanAPI:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': DHAN_ACCESS_TOKEN,
            'client-id': DHAN_CLIENT_ID
        }
        self.last_call_time = 0
    
    def _rate_limit_delay(self):
        current_time = time.time()
        time_diff = current_time - self.last_call_time
        if time_diff < 1.0:
            time.sleep(1.0 - time_diff)
        self.last_call_time = time.time()
    
    def get_intraday_data(self, interval="5"):
        self._rate_limit_delay()
        url = "https://api.dhan.co/v2/charts/intraday"
        ist = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist)
        start_date = end_date - timedelta(days=1)
        
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
        except: return None
    
    def get_ltp_data(self):
        self._rate_limit_delay()
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = {NIFTY_SEG: [NIFTY_SCRIP]}
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            return response.json() if response.status_code == 200 else None
        except: return None

def get_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG, "Expiry": expiry}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except: return None

def get_expiry_list():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {'access-token': DHAN_ACCESS_TOKEN, 'client-id': DHAN_CLIENT_ID, 'Content-Type': 'application/json'}
    payload = {"UnderlyingScrip": NIFTY_SCRIP, "UnderlyingSeg": NIFTY_SEG}
    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json() if response.status_code == 200 else None
    except: return None

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def process_candle_data(data):
    if not data or 'open' not in data: return pd.DataFrame()
    
    df = pd.DataFrame({
        'timestamp': data['timestamp'], 'open': data['open'], 'high': data['high'],
        'low': data['low'], 'close': data['close'], 'volume': data['volume']
    })
    
    ist = pytz.timezone('Asia/Kolkata')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
    return df

def calculate_options_volume_profile(option_data, current_price, bins=20):
    if not option_data or 'oc' not in option_data:
        return []
    
    volume_by_strike = {}
    total_option_volume = 0
    
    for strike, strike_data in option_data['oc'].items():
        try:
            strike_price = float(strike)
            ce_volume = 0
            pe_volume = 0
            
            if 'ce' in strike_data and 'volume' in strike_data['ce']:
                ce_volume = strike_data['ce'].get('volume', 0) or 0
            if 'pe' in strike_data and 'volume' in strike_data['pe']:
                pe_volume = strike_data['pe'].get('volume', 0) or 0
            
            total_volume = ce_volume + pe_volume
            if total_volume > 0:
                volume_by_strike[strike_price] = total_volume
                total_option_volume += total_volume
                
        except (ValueError, TypeError):
            continue
    
    if not volume_by_strike or total_option_volume == 0:
        return []
    
    strikes = sorted(volume_by_strike.keys())
    price_low = min(strikes)
    price_high = max(strikes)
    
    price_bins = np.linspace(price_low, price_high, bins + 1)
    volume_hist = np.zeros(bins)
    
    for strike, volume in volume_by_strike.items():
        bin_idx = np.digitize(strike, price_bins) - 1
        bin_idx = max(0, min(bins - 1, bin_idx))
        volume_hist[bin_idx] += volume
    
    if volume_hist.sum() == 0:
        return []
    
    poc_idx = np.argmax(volume_hist)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
    
    total_volume = volume_hist.sum()
    sorted_indices = np.argsort(volume_hist)[::-1]
    cumulative_volume = 0
    value_area_indices = []
    
    for bin_idx in sorted_indices:
        cumulative_volume += volume_hist[bin_idx]
        value_area_indices.append(bin_idx)
        if cumulative_volume >= 0.7 * total_volume:
            break
    
    if value_area_indices:
        va_high = max([price_bins[i + 1] for i in value_area_indices])
        va_low = min([price_bins[i] for i in value_area_indices])
    else:
        va_high = price_high
        va_low = price_low
    
    return [{
        "poc": poc_price,
        "va_high": va_high,
        "va_low": va_low,
        "volume_hist": volume_hist,
        "price_bins": price_bins,
        "total_volume": total_volume,
        "strike_count": len(volume_by_strike)
    }]

def get_volume_profile_insights(profiles, current_price):
    if not profiles or current_price is None: return {}
    
    profile = profiles[0]
    insights = {
        "poc": profile["poc"], "va_high": profile["va_high"], "va_low": profile["va_low"],
        "total_volume": profile["total_volume"], "bias": "NEUTRAL", "strength": "WEAK"
    }
    
    poc_diff = current_price - insights["poc"]
    
    if abs(poc_diff) < 5:
        insights["price_vs_poc"] = f"AT POC (±{poc_diff:+.1f})"
        insights["bias"] = "NEUTRAL"
    elif poc_diff > 0:
        insights["price_vs_poc"] = f"ABOVE POC (+{poc_diff:.1f})"
        insights["bias"] = "BULLISH"
    else:
        insights["price_vs_poc"] = f"BELOW POC ({poc_diff:.1f})"
        insights["bias"] = "BEARISH"
    
    if current_price > insights["va_high"]:
        insights["price_vs_va"] = f"ABOVE VA (+{current_price - insights['va_high']:.1f})"
        if insights["bias"] == "BULLISH": insights["strength"] = "STRONG"
    elif current_price < insights["va_low"]:
        insights["price_vs_va"] = f"BELOW VA (-{insights['va_low'] - current_price:.1f})"
        if insights["bias"] == "BEARISH": insights["strength"] = "STRONG"
    else:
        insights["price_vs_va"] = "INSIDE VALUE AREA"
        insights["strength"] = "MODERATE" if insights["bias"] != "NEUTRAL" else "WEAK"
    
    return insights

def get_pivots(df, timeframe="5", length=4):
    if df.empty: return []
    
    rule_map = {"5": "5min", "10": "10min", "15": "15min"}
    rule = rule_map.get(timeframe, "5min")
    
    df_temp = df.set_index('datetime')
    try:
        resampled = df_temp.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if len(resampled) < length * 2 + 1: return []
        
        max_vals = resampled['high'].rolling(window=length*2+1, center=True).max()
        min_vals = resampled['low'].rolling(window=length*2+1, center=True).min()
        
        pivots = []
        for timestamp, value in resampled['high'][resampled['high'] == max_vals].items():
            pivots.append({'type': 'high', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        
        for timestamp, value in resampled['low'][resampled['low'] == min_vals].items():
            pivots.append({'type': 'low', 'timeframe': timeframe, 'timestamp': timestamp, 'value': value})
        
        return pivots
    except: return []

def create_chart(df, vp_insights, title):
    if df.empty: return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                       row_heights=[0.8, 0.2])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Nifty', increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
        showlegend=False
    ), row=1, col=1)
    
    # Volume Profile overlays
    if vp_insights:
        # POC line
        fig.add_hline(y=vp_insights["poc"], line_dash="dash", line_color=VP_CONFIG["POC_COLOR"],
                     line_width=2, annotation_text=f"POC: {vp_insights['poc']:.1f}",
                     annotation_position="top right", row=1, col=1)
        
        # Value Area
        fig.add_hrect(y0=vp_insights["va_low"], y1=vp_insights["va_high"],
                     fillcolor="rgba(255, 215, 0, 0.1)", line_color="rgba(255, 215, 0, 0.3)",
                     row=1, col=1)
    
    # Pivot levels
    if len(df) > 30:
        for tf, color in zip(["5", "10", "15"], ["#ff9900", "#ff44ff", "#4444ff"]):
            try:
                pivots = get_pivots(df, tf)
                for pivot in pivots[-2:]:
                    fig.add_hline(y=pivot['value'], line_dash="dot", line_color=color,
                                line_width=1, opacity=0.7, row=1, col=1)
            except: continue
    
    # Volume bars (even if zero, for layout consistency)
    volume_colors = ['#00ff88' if c >= o else '#ff4444' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], marker_color=volume_colors,
                        opacity=0.7, showlegend=False), row=2, col=1)
    
    fig.update_layout(title=title, template='plotly_dark', height=600,
                     xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def analyze_options(expiry):
    option_data = get_option_chain(expiry)
    if not option_data or 'data' not in option_data: return None, None
    
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
        
        ce_oi = row['openInterest_CE']
        pe_oi = row['openInterest_PE']
        level = "Support" if pe_oi > 1.12 * ce_oi else "Resistance" if ce_oi > 1.12 * pe_oi else "Neutral"
        
        bias_results.append({
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": level,
            "ChgOI_Bias": chg_oi_bias,
            "Volume_Bias": volume_bias,
            "PCR": round(pe_oi / ce_oi if ce_oi > 0 else 0, 2),
            "changeinOpenInterest_CE": row['changeinOpenInterest_CE'],
            "changeinOpenInterest_PE": row['changeinOpenInterest_PE']
        })
    
    return underlying, pd.DataFrame(bias_results)

def check_signals(option_data, current_price, vp_insights):
    if not option_data or not current_price: return
    
    atm_data = option_data[option_data['Zone'] == 'ATM']
    if atm_data.empty: return
    
    row = atm_data.iloc[0]
    
    # Volume Profile confirmation
    vp_confirmation = ""
    if vp_insights:
        if vp_insights["bias"] in ["BULLISH", "BEARISH"]:
            vp_confirmation = f"VP: {vp_insights['price_vs_poc']} | {vp_insights['strength']} {vp_insights['bias']}"
        else:
            vp_confirmation = f"VP: {vp_insights['price_vs_poc']} | {vp_insights['bias']}"
    
    # Signal logic (simplified)
    if row['Level'] == 'Support' and row['ChgOI_Bias'] == 'Bullish':
        message = f"""
🚨 NIFTY CALL SIGNAL 🚨

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']} - {row['Level']}
💰 {vp_confirmation}
📊 ChgOI: {row['ChgOI_Bias']} | Volume: {row['Volume_Bias']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success("🔔 CALL signal sent!")
    
    elif row['Level'] == 'Resistance' and row['ChgOI_Bias'] == 'Bearish':
        message = f"""
🚨 NIFTY PUT SIGNAL 🚨

📍 Spot: ₹{current_price:.2f}
🎯 ATM: {row['Strike']} - {row['Level']}
💰 {vp_confirmation}
📊 ChgOI: {row['ChgOI_Bias']} | Volume: {row['Volume_Bias']}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')}
"""
        send_telegram(message)
        st.success("🔔 PUT signal sent!")

def main():
    st.title("📈 Nifty Analyzer with Volume Profile")
    
    ist = pytz.timezone('Asia/Kolkata')
    if not is_market_hours():
        st.warning(f"Market closed. Time: {datetime.now(ist).strftime('%H:%M:%S IST')}")
    
    st.sidebar.header("Settings")
    interval = st.sidebar.selectbox("Timeframe", ["1", "3", "5", "10", "15"], index=2)
    proximity = st.sidebar.slider("Signal Proximity", 1, 20, 5)
    enable_signals = st.sidebar.checkbox("Enable Signals", value=True)
    VP_CONFIG["BINS"] = st.sidebar.slider("VP Bins", 10, 30, 20)
    
    manual_refresh = st.sidebar.checkbox("Manual Refresh", value=True)
    if manual_refresh:
        refresh_data = st.sidebar.button("🔄 Refresh Data")
    else:
        refresh_data = True
    
    api = DhanAPI()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chart with Options Volume Profile")
        
        if refresh_data:
            data = api.get_intraday_data(interval)
            df = process_candle_data(data) if data else pd.DataFrame()
            st.session_state.chart_data = df
        else:
            df = st.session_state.get('chart_data', pd.DataFrame())
            if df.empty:
                st.info("Click 'Refresh Data'")
                return
        
        current_price = df['close'].iloc[-1] if not df.empty else None
        
        # Get Volume Profile from options
        vp_insights = None
        if current_price:
            try:
                expiry_data = get_expiry_list()
                if expiry_data and 'data' in expiry_data:
                    selected_expiry = expiry_data['data'][0]
                    option_data = get_option_chain(selected_expiry)
                    if option_data and 'data' in option_data:
                        profiles = calculate_options_volume_profile(option_data['data'], current_price, VP_CONFIG["BINS"])
                        if profiles:
                            vp_insights = get_volume_profile_insights(profiles, current_price)
            except: pass
        
        # Display metrics
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
        
        # Volume Profile display
        if vp_insights:
            st.subheader("💰 Options Volume Profile")
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
            
            st.info(f"📍 Position: {vp_insights['price_vs_poc']} | {vp_insights['price_vs_va']}")
            st.info(f"💪 Strength: {vp_insights['strength']} | Volume: {vp_insights['total_volume']:,.0f}")
        
        if not df.empty:
            fig = create_chart(df, vp_insights, f"Nifty {interval}min")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("Options Analysis")
        
        expiry_data = get_expiry_list()
        if expiry_data and 'data' in expiry_data:
            selected_expiry = st.selectbox("Expiry", expiry_data['data'])
            underlying_price, option_summary = analyze_options(selected_expiry)
            
            if underlying_price and option_summary is not None:
                st.info(f"Spot: ₹{underlying_price:.2f}")
                st.dataframe(option_summary, use_container_width=True)
                
                if enable_signals and not df.empty and is_market_hours():
                    check_signals(option_summary, underlying_price, vp_insights)
            else:
                st.error("Options data unavailable")
    
    if st.sidebar.button("Test Telegram"):
        test_message = f"🔔 Test from Nifty Analyzer - {datetime.now(ist).strftime('%H:%M:%S IST')}"
        send_telegram(test_message)
        st.sidebar.success("Test sent!")

if __name__ == "__main__":
    main()
