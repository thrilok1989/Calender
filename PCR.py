# full_nifty_analyzer_with_enhancements.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone
import plotly.graph_objects as go
import io
import json

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer + Enhancements", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 minutes

# === Telegram Config (consider env var in production) ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

# === Session state init ===
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'call_log_book' not in st.session_state:
    st.session_state.call_log_book = []
if 'export_data' not in st.session_state:
    st.session_state.export_data = False
if 'support_zone' not in st.session_state:
    st.session_state.support_zone = (None, None)
if 'resistance_zone' not in st.session_state:
    st.session_state.resistance_zone = (None, None)

# PCR defaults
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 2.0
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.4
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal", "VIX"])

# Enhancements defaults
st.session_state.setdefault('enh_enable_sudden', True)
st.session_state.setdefault('enh_oi_threshold_pct', 25)
st.session_state.setdefault('enh_vol_threshold_pct', 40)
st.session_state.setdefault('enh_enable_iv', True)
st.session_state.setdefault('enh_iv_window', 5)
st.session_state.setdefault('enh_enable_maxpain', True)
st.session_state.setdefault('iv_history', [])

# === Utility functions (your original ones) ===
def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'CE':
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma)/(2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2))/365
        rho = (K * T * math.exp(-r * T) * norm.cdf(d2))/100
    else:
        delta = -norm.cdf(-d1)
        theta = (-(S * norm.pdf(d1) * sigma)/(2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2))/365
        rho = (-K * T * math.exp(-r * T) * norm.cdf(-d2))/100
    gamma = norm.pdf(d1)/(S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)/100
    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

def final_verdict(score):
    if score >= 4: return "Strong Bullish"
    elif score >= 2: return "Bullish"
    elif score <= -4: return "Strong Bearish"
    elif score <= -2: return "Bearish"
    return "Neutral"

def delta_volume_bias(price, volume, chg_oi):
    if price > 0 and volume > 0 and chg_oi > 0: return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0: return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0: return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0: return "Bearish"
    return "Neutral"

def determine_level(row):
    try:
        if row['openInterest_PE'] > 1.12 * row['openInterest_CE']: return "Support"
        elif row['openInterest_CE'] > 1.12 * row['openInterest_PE']: return "Resistance"
    except Exception:
        return "Neutral"
    return "Neutral"

def is_in_zone(spot, strike, level):
    if level in ["Support", "Resistance"]:
        return strike - 10 <= spot <= strike + 10
    return False

def get_support_resistance_zones(df, spot):
    support_strikes = df[df['Level'] == "Support"]['strikePrice'].tolist()
    resistance_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()
    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistance_strikes if r >= spot])[:2]
    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports[0]) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances[0], nearest_resistances[0]) if nearest_resistances else (None, None)
    return support_zone, resistance_zone

def display_enhanced_trade_log():
    if not st.session_state.trade_log:
        st.info("No trades logged yet")
        return
    st.markdown("### 📜 Enhanced Trade Log")
    df_trades = pd.DataFrame(st.session_state.trade_log)
    if 'Current_Price' not in df_trades.columns:
        df_trades['Current_Price'] = df_trades['LTP'] * np.random.uniform(0.8, 1.3, len(df_trades))
        df_trades['Unrealized_PL'] = (df_trades['Current_Price'] - df_trades['LTP']) * 75
        df_trades['Status'] = df_trades['Unrealized_PL'].apply(
            lambda x: '🟢 Profit' if x > 0 else '🔴 Loss' if x < -100 else '🟡 Breakeven'
        )
    def color_pnl(row):
        colors = []
        for col in row.index:
            if col == 'Unrealized_PL':
                if row[col] > 0:
                    colors.append('background-color: #90EE90; color: black')
                elif row[col] < -100:
                    colors.append('background-color: #FFB6C1; color: black')
                else:
                    colors.append('background-color: #FFFFE0; color: black')
            else:
                colors.append('')
        return colors
    styled_trades = df_trades.style.apply(color_pnl, axis=1)
    st.dataframe(styled_trades, use_container_width=True)
    total_pl = df_trades['Unrealized_PL'].sum()
    win_rate = len(df_trades[df_trades['Unrealized_PL'] > 0]) / len(df_trades) * 100
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total P&L", f"₹{total_pl:,.0f}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total Trades", len(df_trades))

def create_export_data(df_summary, trade_log, spot_price):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Option_Chain_Summary', index=False)
        if trade_log:
            pd.DataFrame(trade_log).to_excel(writer, sheet_name='Trade_Log', index=False)
        if not st.session_state.pcr_history.empty:
            st.session_state.pcr_history.to_excel(writer, sheet_name='PCR_History', index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_analysis_{timestamp}.xlsx"
    return output.getvalue(), filename

def handle_export_data(df_summary, spot_price):
    if 'export_data' in st.session_state and st.session_state.export_data:
        try:
            excel_data, filename = create_export_data(df_summary, st.session_state.trade_log, spot_price)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.success("✅ Export ready! Click the download button above.")
            st.session_state.export_data = False
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            st.session_state.export_data = False

def plot_price_with_sr():
    price_df = st.session_state['price_data'].copy()
    if price_df.empty or price_df['Spot'].isnull().all():
        st.info("Not enough data to show price action chart yet.")
        return
    price_df['Time'] = pd.to_datetime(price_df['Time'])
    support_zone = st.session_state.get('support_zone', (None, None))
    resistance_zone = st.session_state.get('resistance_zone', (None, None))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df['Time'],
        y=price_df['Spot'],
        mode='lines+markers',
        name='Spot Price',
        line=dict(color='blue', width=2)
    ))
    if all(support_zone) and None not in support_zone:
        fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1,
                      y0=support_zone[0], y1=support_zone[1],
                      fillcolor="rgba(0,255,0,0.08)", line=dict(width=0), layer="below")
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[0], support_zone[0]],
            mode='lines', name='Support Low', line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[1], support_zone[1]],
            mode='lines', name='Support High', line=dict(color='green', dash='dot')
        ))
    if all(resistance_zone) and None not in resistance_zone:
        fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1,
                      y0=resistance_zone[0], y1=resistance_zone[1],
                      fillcolor="rgba(255,0,0,0.08)", line=dict(width=0), layer="below")
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[0], resistance_zone[0]],
            mode='lines', name='Resistance Low', line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[1], resistance_zone[1]],
            mode='lines', name='Resistance High', line=dict(color='red', dash='dot')
        ))
    fig.update_layout(
        title="Nifty Spot Price Action with Support & Resistance",
        xaxis_title="Time",
        yaxis_title="Spot Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def auto_update_call_log(current_price):
    for call in st.session_state.call_log_book:
        if call["Status"] != "Active":
            continue
        if call["Type"] == "CE":
            if current_price >= max(call["Targets"].values()):
                call["Status"] = "Hit Target"
                call["Hit_Target"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
            elif current_price <= call["Stoploss"]:
                call["Status"] = "Hit Stoploss"
                call["Hit_Stoploss"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
        elif call["Type"] == "PE":
            if current_price <= min(call["Targets"].values()):
                call["Status"] = "Hit Target"
                call["Hit_Target"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
            elif current_price >= call["Stoploss"]:
                call["Status"] = "Hit Stoploss"
                call["Hit_Stoploss"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price

def display_call_log_book():
    st.markdown("### 📚 Call Log Book")
    if not st.session_state.call_log_book:
        st.info("No calls have been made yet.")
        return
    df_log = pd.DataFrame(st.session_state.call_log_book)
    st.dataframe(df_log, use_container_width=True)
    if st.button("Download Call Log Book as CSV"):
        st.download_button(
            label="Download CSV",
            data=df_log.to_csv(index=False).encode(),
            file_name="call_log_book.csv",
            mime="text/csv"
        )

# ===================== Enhancements: sudden OI/Volume, Max Pain, IV trend/skew =====================
def render_enhancements_ui():
    st.markdown("### ⚡ Intraday Enhancements & Signals")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.enh_enable_sudden = st.checkbox(
            "Enable Sudden OI/Volume Alerts", value=st.session_state.get('enh_enable_sudden', True)
        )
        st.session_state.enh_oi_threshold_pct = st.number_input(
            "OI spike % threshold", min_value=1, max_value=1000, value=st.session_state.get('enh_oi_threshold_pct', 25)
        )
        st.session_state.enh_vol_threshold_pct = st.number_input(
            "Volume spike % threshold", min_value=1, max_value=1000, value=st.session_state.get('enh_vol_threshold_pct', 40)
        )
    with col2:
        st.session_state.enh_enable_iv = st.checkbox(
            "Enable IV Trend & Skew Analysis", value=st.session_state.get('enh_enable_iv', True)
        )
        st.session_state.enh_iv_window = st.number_input(
            "IV trend window (ticks)", min_value=2, max_value=50, value=st.session_state.get('enh_iv_window', 5)
        )
        st.session_state.enh_enable_maxpain = st.checkbox(
            "Enable Max Pain Analysis", value=st.session_state.get('enh_enable_maxpain', True)
        )

def detect_sudden_spikes(df, prev_df, oi_threshold_pct=25, vol_threshold_pct=40):
    alerts = []
    if prev_df is None:
        return alerts
    merged = df.merge(prev_df, on=['strikePrice','type'], suffixes=('', '_prev'))
    for _, row in merged.iterrows():
        try:
            if row['openInterest_prev'] > 0:
                oi_change = (row['openInterest'] - row['openInterest_prev'])/row['openInterest_prev']*100
                if oi_change >= oi_threshold_pct:
                    alerts.append(f"Sudden OI spike {oi_change:.1f}% on {row['strikePrice']} {row['type']}")
        except Exception:
            pass
        try:
            if row['totalTradedVolume_prev'] > 0:
                vol_change = (row['totalTradedVolume'] - row['totalTradedVolume_prev'])/row['totalTradedVolume_prev']*100
                if vol_change >= vol_threshold_pct:
                    alerts.append(f"Sudden Volume spike {vol_change:.1f}% on {row['strikePrice']} {row['type']}")
        except Exception:
            pass
    return alerts

def calculate_max_pain(df):
    strikes = df['strikePrice'].unique()
    pain = {}
    for strike in strikes:
        # CE losses if expiry ends below strike
        ce_loss = ((df[(df['strikePrice']>strike)&(df['type']=='CE')]['openInterest']) * (df[(df['strikePrice']>strike)&(df['type']=='CE')]['strikePrice'] - strike)).sum()
        # PE losses if expiry ends above strike
        pe_loss = ((df[(df['strikePrice']<strike)&(df['type']=='PE')]['openInterest']) * (strike - df[(df['strikePrice']<strike)&(df['type']=='PE')]['strikePrice'])).sum()
        pain[strike] = ce_loss + pe_loss
    if not pain:
        return None
    max_pain = min(pain, key=pain.get)
    return max_pain

def analyze_iv_trend(df, iv_window=5):
    # store average IV per type in session history
    avg = df.groupby('type')['impliedVolatility'].mean().to_dict()
    hist = st.session_state.get('iv_history', [])
    hist.append(avg)
    if len(hist) > iv_window:
        hist.pop(0)
    st.session_state['iv_history'] = hist
    if len(hist) < 2:
        return None
    iv_trend = {
        'CE_trend': hist[-1].get('CE', 0) - hist[0].get('CE', 0),
        'PE_trend': hist[-1].get('PE', 0) - hist[0].get('PE', 0),
        'Skew': hist[-1].get('CE', 0) - hist[-1].get('PE', 0)
    }
    return iv_trend

def combine_oi_iv_signals(df):
    signals = []
    # Precompute averages to compare
    iv_mean = df['impliedVolatility'].mean() if 'impliedVolatility' in df.columns else 0
    for _, row in df.iterrows():
        try:
            if row.get('changeinOpenInterest', 0) > 0 and row.get('impliedVolatility', 0) > iv_mean:
                signals.append((row['strikePrice'], row['type'], 'Long Build-up'))
            elif row.get('changeinOpenInterest', 0) > 0 and row.get('impliedVolatility', 0) < iv_mean:
                signals.append((row['strikePrice'], row['type'], 'Short Build-up'))
            elif row.get('changeinOpenInterest', 0) < 0 and row.get('impliedVolatility', 0) < iv_mean:
                signals.append((row['strikePrice'], row['type'], 'Long Unwinding'))
            elif row.get('changeinOpenInterest', 0) < 0 and row.get('impliedVolatility', 0) > iv_mean:
                signals.append((row['strikePrice'], row['type'], 'Short Covering'))
        except Exception:
            continue
    return signals

def check_banknifty_alignment(nifty_bias, banknifty_df):
    if banknifty_df is None:
        return "No Data"
    # very lightweight: sum CE OI - sum PE OI
    bank_bias = banknifty_df['CE_OI'].sum() - banknifty_df['PE_OI'].sum()
    if (nifty_bias>0 and bank_bias>0) or (nifty_bias<0 and bank_bias<0):
        return "Aligned"
    else:
        return "Diverging"

def run_enhancements(session, df_long, df_summary, underlying, now, send_telegram_message, prev_df=None, banknifty_df=None):
    results = {}
    if st.session_state.get('enh_enable_sudden'):
        results['sudden_alerts'] = detect_sudden_spikes(df_long, prev_df,
                                                       st.session_state.enh_oi_threshold_pct,
                                                       st.session_state.enh_vol_threshold_pct)
        for alert in results['sudden_alerts']:
            send_telegram_message(f"⚡ {alert}")
    if st.session_state.get('enh_enable_maxpain'):
        results['max_pain'] = calculate_max_pain(df_long)
    if st.session_state.get('enh_enable_iv'):
        results['iv_trend'] = analyze_iv_trend(df_long, st.session_state.enh_iv_window)
    results['oi_iv_signals'] = combine_oi_iv_signals(df_long)
    results['banknifty_alignment'] = check_banknifty_alignment(df_summary.get('bias', 0) if isinstance(df_summary, dict) else 0, banknifty_df)
    return results

# ===================== Main analyze() =====================
def analyze():
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("19:40", "%H:%M").time()

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            return

        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        try:
            session.get("https://www.nseindia.com", timeout=5)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to establish NSE session: {e}")
            return

        # VIX
        vix_value = 11
        try:
            vix_url = "https://www.nseindia.com/api/equity-stockIndices?index=INDIA%20VIX"
            vix_response = session.get(vix_url, timeout=10)
            vix_response.raise_for_status()
            vix_data = vix_response.json()
            vix_value = vix_data['data'][0]['lastPrice']
        except Exception:
            vix_value = 11

        if vix_value > 12:
            st.session_state.pcr_threshold_bull = 2.0
            st.session_state.pcr_threshold_bear = 0.4
            volatility_status = "High Volatility"
        else:
            st.session_state.pcr_threshold_bull = 1.2
            st.session_state.pcr_threshold_bear = 0.7
            volatility_status = "Low Volatility"

        # Option chain
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"❌ Failed to get option chain data: {e}")
            return

        if not data or 'records' not in data:
            st.error("❌ Empty or invalid response from NSE API")
            return

        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']

        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.markdown(f"### 📊 VIX: {vix_value} ({volatility_status}) | PCR Thresholds: Bull >{st.session_state.pcr_threshold_bull} | Bear <{st.session_state.pcr_threshold_bear}")

        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        today = datetime.now(timezone("Asia/Kolkata"))
        T = max((expiry_date - today).days, 1) / 365
        r = 0.06

        # Build CE and PE lists
        calls, puts = [], []
        for item in records:
            if 'CE' in item and item['CE']['expiryDate'] == expiry:
                ce = item['CE']
                if ce.get('impliedVolatility', 0) > 0:
                    greeks = calculate_greeks('CE', underlying, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                    ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                calls.append(ce)
            if 'PE' in item and item['PE']['expiryDate'] == expiry:
                pe = item['PE']
                if pe.get('impliedVolatility', 0) > 0:
                    greeks = calculate_greeks('PE', underlying, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                    pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                puts.append(pe)

        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        if df_ce.empty or df_pe.empty:
            st.error("❌ CE or PE frame empty - cannot proceed")
            return
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

        # ATM filtering
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        # Bias scoring (your original approach)
        weights = {'ChgOI_Bias': 1.5,'Volume_Bias': 1.0,'Gamma_Bias': 1.2,'AskQty_Bias': 0.8,'BidQty_Bias': 0.8,'IV_Bias': 1.0,'DVP_Bias': 1.5}
        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue
            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                "BidQty_Bias": "Bearish" if row['bidQty_PE'] > row['bidQty_CE'] else "Bullish",
                "IV_Bias": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row['lastPrice_CE'] - row['lastPrice_PE'],
                    row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                    row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
                )
            }
            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)
            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)

        # PCR calculation & merge
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE']],
            left_on='Strike',
            right_on='strikePrice',
            how='left'
        )
        df_summary['PCR'] = np.where(df_summary['openInterest_CE'] == 0, 0, (df_summary['openInterest_PE'] / df_summary['openInterest_CE']).round(2))
        df_summary['PCR_Signal'] = np.where(
            df_summary['PCR'] > st.session_state.pcr_threshold_bull,
            "Bullish",
            np.where(df_summary['PCR'] < st.session_state.pcr_threshold_bear, "Bearish", "Neutral")
        )
        def color_pcr(val):
            if val > st.session_state.pcr_threshold_bull:
                return 'background-color: #90EE90; color: black'
            elif val < st.session_state.pcr_threshold_bear:
                return 'background-color: #FFB6C1; color: black'
            else:
                return 'background-color: #FFFFE0; color: black'
        styled_df = df_summary.style.applymap(color_pcr, subset=['PCR'])
        df_summary = df_summary.drop(columns=['strikePrice'])

        # Record PCR history
        for _, row in df_summary.iterrows():
            new_pcr_data = pd.DataFrame({
                "Time": [now.strftime("%H:%M:%S")],
                "Strike": [row['Strike']],
                "PCR": [row['PCR']],
                "Signal": [row['PCR_Signal']],
                "VIX": [vix_value]
            })
            st.session_state.pcr_history = pd.concat([st.session_state.pcr_history, new_pcr_data], ignore_index=True)

        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)
        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone

        # Update spot price history
        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        # Build a long-format df for enhancements (columns: strikePrice, type, openInterest, totalTradedVolume, impliedVolatility, changeinOpenInterest, lastPrice)
        ce_long = df[['strikePrice',
                      'openInterest_CE', 'totalTradedVolume_CE', 'impliedVolatility_CE', 'changeinOpenInterest_CE', 'lastPrice_CE']].copy()
        ce_long.columns = ['strikePrice', 'openInterest', 'totalTradedVolume', 'impliedVolatility', 'changeinOpenInterest', 'lastPrice']
        ce_long['type'] = 'CE'
        pe_long = df[['strikePrice',
                      'openInterest_PE', 'totalTradedVolume_PE', 'impliedVolatility_PE', 'changeinOpenInterest_PE', 'lastPrice_PE']].copy()
        pe_long.columns = ['strikePrice', 'openInterest', 'totalTradedVolume', 'impliedVolatility', 'changeinOpenInterest', 'lastPrice']
        pe_long['type'] = 'PE'
        long_df = pd.concat([ce_long, pe_long], ignore_index=True)

        # Prev snapshot for sudden spike detection
        prev_snapshot = st.session_state.get('prev_option_snapshot')  # None first time

        # Run enhancements
        render_enhancements_ui()
        enhancements = run_enhancements(session, long_df, df_summary.to_dict(orient='records') if not df_summary.empty else {}, underlying, now, send_telegram_message, prev_df=prev_snapshot, banknifty_df=None)

        # Save snapshot for next tick
        st.session_state['prev_option_snapshot'] = long_df.copy(deep=True)

        # Format zones
        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        # Signal generation (existing logic) - simplified to use df_summary and bias_results
        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False
        last_trade = st.session_state.trade_log[-1] if st.session_state.trade_log else None
        if last_trade and not (last_trade.get("TargetHit", False) or last_trade.get("SLHit", False)):
            pass
        else:
            for row in bias_results:
                if not is_in_zone(underlying, row['Strike'], row['Level']):
                    continue
                # get PCR info
                pcr_row = df_summary[df_summary['Strike'] == row['Strike']].iloc[0]
                pcr_signal = pcr_row['PCR_Signal']
                pcr_value = pcr_row['PCR']

                atm_chgoi_bias = atm_row['ChgOI_Bias'] if atm_row is not None else None
                atm_askqty_bias = atm_row['AskQty_Bias'] if atm_row is not None else None

                if st.session_state.use_pcr_filter:
                    if (row['Level'] == "Support" and total_score >= 4 
                        and "Bullish" in market_view
                        and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                        and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)
                        and pcr_signal == "Bullish"):
                        option_type = 'CE'
                    elif (row['Level'] == "Resistance" and total_score <= -4 
                          and "Bearish" in market_view
                          and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                          and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                          and pcr_signal == "Bearish"):
                        option_type = 'PE'
                    else:
                        continue
                else:
                    if (row['Level'] == "Support" and total_score >= 4 
                        and "Bullish" in market_view
                        and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                        and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)):
                        option_type = 'CE'
                    elif (row['Level'] == "Resistance" and total_score <= -4 
                          and "Bearish" in market_view
                          and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                          and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)):
                        option_type = 'PE'
                    else:
                        continue

                # Option details from df (CE/PE)
                ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
                iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
                target = round(ltp * (1 + iv / 100), 2)
                stop_loss = round(ltp * 0.8, 2)

                atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
                suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"

                send_telegram_message(
                    f"VIX: {vix_value} ({volatility_status})\n"
                    f"PCR: {pcr_value} ({pcr_signal})\n"
                    f"Thresholds: Bull>{st.session_state.pcr_threshold_bull} Bear<{st.session_state.pcr_threshold_bear}\n"
                    f"📍 Spot: {underlying}\n"
                    f"🔹 {atm_signal}\n"
                    f"{suggested_trade}\n"
                    f"Bias Score: {total_score} ({market_view})\n"
                    f"Level: {row['Level']}\n"
                    f"📉 Support Zone: {support_str}\n"
                    f"📈 Resistance Zone: {resistance_str}"
                )

                st.session_state.trade_log.append({
                    "Time": now.strftime("%H:%M:%S"),
                    "Strike": row['Strike'],
                    "Type": option_type,
                    "LTP": ltp,
                    "Target": target,
                    "SL": stop_loss,
                    "TargetHit": False,
                    "SLHit": False,
                    "VIX": vix_value,
                    "PCR_Value": pcr_value,
                    "PCR_Signal": pcr_signal,
                    "PCR_Thresholds": f"Bull>{st.session_state.pcr_threshold_bull} Bear<{st.session_state.pcr_threshold_bear}"
                })

                signal_sent = True
                break

        # Display
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")

        # Show enhancements results
        with st.expander("⚡ Enhancements: Sudden OI/Vol, IV Trend, Max Pain & OI+IV Signals"):
            st.write("Enhancements results (alerts sent via Telegram if any):")
            st.write(enhancements)
            if enhancements.get('sudden_alerts'):
                for a in enhancements['sudden_alerts']:
                    st.warning(a)
            if enhancements.get('max_pain') is not None:
                st.info(f"Max Pain: {enhancements['max_pain']}")
            if enhancements.get('iv_trend') is not None:
                st.write("IV Trend & Skew:", enhancements['iv_trend'])
            if enhancements.get('oi_iv_signals'):
                st.dataframe(pd.DataFrame(enhancements['oi_iv_signals'], columns=['Strike','Type','Signal']))

        # Plot price action
        plot_price_with_sr()

        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")

        # Option Chain Summary
        with st.expander("📊 Option Chain Summary"):
            st.info(f"""
            ℹ️ PCR Interpretation (VIX: {vix_value}):
            - >{st.session_state.pcr_threshold_bull} = Bullish
            - <{st.session_state.pcr_threshold_bear} = Bearish
            - Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'}
            """)
            st.dataframe(styled_df)

        # Trade Log
        if st.session_state.trade_log:
            st.markdown("### 📜 Trade Log")
            st.dataframe(pd.DataFrame(st.session_state.trade_log))

        # Enhanced Features / PCR Config UI
        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")
        st.markdown("### 🧮 PCR Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pcr_threshold_bull = st.number_input(
                "Bullish PCR Threshold (>)",
                min_value=1.0, max_value=5.0,
                value=st.session_state.pcr_threshold_bull,
                step=0.1
            )
        with col2:
            st.session_state.pcr_threshold_bear = st.number_input(
                "Bearish PCR Threshold (<)",
                min_value=0.1, max_value=1.0,
                value=st.session_state.pcr_threshold_bear,
                step=0.1
            )
        with col3:
            st.session_state.use_pcr_filter = st.checkbox(
                "Enable PCR Filtering",
                value=st.session_state.use_pcr_filter
            )

        # PCR History
        with st.expander("📈 PCR History"):
            if not st.session_state.pcr_history.empty:
                pcr_pivot = st.session_state.pcr_history.pivot_table(index='Time', columns='Strike', values='PCR', aggfunc='last')
                st.line_chart(pcr_pivot)
                st.dataframe(st.session_state.pcr_history)
            else:
                st.info("No PCR history recorded yet")

        # Enhanced Trade Log
        display_enhanced_trade_log()

        # Export
        st.markdown("---")
        st.markdown("### 📥 Data Export")
        if st.button("Prepare Excel Export"):
            st.session_state.export_data = True
        handle_export_data(df_summary, underlying)

        # Call Log Book
        st.markdown("---")
        display_call_log_book()

        # Auto update call log with current price
        auto_update_call_log(underlying)

    except json.JSONDecodeError as e:
        st.error("❌ Failed to decode JSON response from NSE API. The market might be closed or the API is unavailable.")
        send_telegram_message("❌ NSE API JSON decode error - Market may be closed")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {e}")
        send_telegram_message(f"❌ Network error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        send_telegram_message(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    analyze()
