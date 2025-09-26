import streamlit as st
import pandas as pd
import requests
import time
import datetime
from supabase import create_client
import plotly.graph_objects as go
import numpy as np

# --- Streamlit Config ---
st.set_page_config(page_title="Nifty Option Screener – Intraday Fresh PCR", layout="wide")

# --- Market Hours Check ---
def is_market_open():
    now_utc = datetime.datetime.utcnow()
    ist_offset = datetime.timedelta(hours=5, minutes=30)
    now_ist = now_utc + ist_offset
    # Weekday: Monday=0, Friday=4
    if now_ist.weekday() > 4:
        return False
    market_open = datetime.time(9, 0)
    market_close = datetime.time(18, 40)
    return market_open <= now_ist.time() <= market_close

if not is_market_open():
    st.warning("⛔ Market is closed. Data fetch only runs Monday-Friday 9:00–15:40 IST.")
    st.stop()

# Strike range selector
strike_range = st.sidebar.slider("📏 Strike Range (±)", 200, 1000, 500, 50)

# --- Auto Refresh ---
def auto_refresh(interval_sec=60):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()
auto_refresh(60)

# --- Telegram Setup ---
def send_telegram_message(message):
    try:
        bot_token = st.secrets["TELEGRAM"]["BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM"]["CHAT_ID"]
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to send Telegram message: {e}")
        return False

# --- Supabase Setup ---
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase = create_client(supabase_url, supabase_key)

# --- Enhanced Log PCR to Supabase (with fresh OI data) ---
def log_pcr(expiry, underlying, atm_strike, pcr, total_ce_oi, total_pe_oi, fresh_pcr, fresh_ce_oi, fresh_pe_oi):
    try:
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "expiry": expiry,
            "underlying": underlying,
            "atm_strike": atm_strike,
            "pcr": pcr,
            "total_ce_oi": int(total_ce_oi),
            "total_pe_oi": int(total_pe_oi),
            "fresh_pcr": fresh_pcr,
            "fresh_ce_oi": int(fresh_ce_oi),
            "fresh_pe_oi": int(fresh_pe_oi)
        }
        supabase.table("nifty_pcr_logs").insert(data).execute()
    except Exception as e:
        st.error(f"Supabase logging failed: {e}")

# --- Fetch Option Chain ---
@st.cache_data(ttl=180)
def fetch_option_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers)
        r = s.get(url, headers=headers, timeout=3)
        return r.json()
    except:
        st.warning("⚠️ Could not fetch data from NSE.")
        return None

# --- Fetch Data ---
data = fetch_option_chain()
if data is None: st.stop()
raw_data = data['records']['data']
expiry_list = data['records']['expiryDates']
underlying = data['records'].get('underlyingValue', 0)
selected_expiry = st.sidebar.selectbox("📅 Select Expiry Date", expiry_list)

# --- CE & PE Data ---
ce_list, pe_list = [], []
for item in raw_data:
    if item.get("expiryDate") == selected_expiry:
        strike = item.get("strikePrice", 0)
        if "CE" in item:
            ce = item["CE"]
            ce_list.append({
                "strikePrice": strike,
                "OI_CE": ce.get("openInterest", 0),
                "Chg_OI_CE": ce.get("changeinOpenInterest", 0),
                "Vol_CE": ce.get("totalTradedVolume", 0),
            })
        if "PE" in item:
            pe = item["PE"]
            pe_list.append({
                "strikePrice": strike,
                "OI_PE": pe.get("openInterest", 0),
                "Chg_OI_PE": pe.get("changeinOpenInterest", 0),
                "Vol_PE": pe.get("totalTradedVolume", 0),
            })

df_ce = pd.DataFrame(ce_list)
df_pe = pd.DataFrame(pe_list)

# --- ATM Strike & Filter ---
atm_strike = min(df_ce['strikePrice'], key=lambda x: abs(x - underlying))
df_ce = df_ce[(df_ce["strikePrice"] >= atm_strike - strike_range) & (df_ce["strikePrice"] <= atm_strike + strike_range)]
df_pe = df_pe[(df_pe["strikePrice"] >= atm_strike - strike_range) & (df_pe["strikePrice"] <= atm_strike + strike_range)]

# --- PCR Calculations (Both Total and Fresh) ---
# Total PCR (your original)
total_ce_oi = df_ce["OI_CE"].sum()
total_pe_oi = df_pe["OI_PE"].sum()
pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
if "previous_pcr" not in st.session_state: st.session_state.previous_pcr = pcr
pcr_change = pcr - st.session_state.previous_pcr
st.session_state.previous_pcr = pcr

# Fresh PCR (based on OI changes)
fresh_ce_oi = df_ce["Chg_OI_CE"].sum()
fresh_pe_oi = df_pe["Chg_OI_PE"].sum()
fresh_pcr = round(fresh_pe_oi / fresh_ce_oi, 2) if fresh_ce_oi > 0 else 0
if "previous_fresh_pcr" not in st.session_state: st.session_state.previous_fresh_pcr = fresh_pcr
fresh_pcr_change = fresh_pcr - st.session_state.previous_fresh_pcr
st.session_state.previous_fresh_pcr = fresh_pcr

# Volume PCR
total_ce_vol = df_ce["Vol_CE"].sum()
total_pe_vol = df_pe["Vol_PE"].sum()
pcr_vol = round(total_pe_vol / total_ce_vol, 2) if total_ce_vol > 0 else 0

# --- Session-wise Fresh OI Tracking ---
def get_current_session():
    now_utc = datetime.datetime.utcnow()
    ist_offset = datetime.timedelta(hours=5, minutes=30)
    now_ist = now_utc + ist_offset
    current_time = now_ist.time()
    
    if datetime.time(9,15) <= current_time < datetime.time(12,30): 
        return "Morning"
    elif datetime.time(12,30) <= current_time < datetime.time(14,30): 
        return "Afternoon"
    elif datetime.time(14,30) <= current_time <= datetime.time(15,40): 
        return "Closing"
    else: 
        return "Off Hours"

current_session = get_current_session()

# Initialize session OI tracking
if "session_oi_start" not in st.session_state:
    st.session_state.session_oi_start = {}
if "last_session" not in st.session_state:
    st.session_state.last_session = current_session

# Reset OI tracking when session changes
if st.session_state.last_session != current_session:
    st.session_state.session_oi_start[current_session] = {
        "ce_oi": total_ce_oi,
        "pe_oi": total_pe_oi,
        "timestamp": datetime.datetime.now()
    }
    st.session_state.last_session = current_session

# Calculate session-wise fresh PCR
session_fresh_pcr = 0
if current_session in st.session_state.session_oi_start:
    start_ce_oi = st.session_state.session_oi_start[current_session]["ce_oi"]
    start_pe_oi = st.session_state.session_oi_start[current_session]["pe_oi"]
    
    session_fresh_ce = total_ce_oi - start_ce_oi
    session_fresh_pe = total_pe_oi - start_pe_oi
    
    session_fresh_pcr = round(session_fresh_pe / session_fresh_ce, 2) if session_fresh_ce > 0 else 0
else:
    # Initialize for current session
    st.session_state.session_oi_start[current_session] = {
        "ce_oi": total_ce_oi,
        "pe_oi": total_pe_oi,
        "timestamp": datetime.datetime.now()
    }

# --- Max Pain calculation ---
def calculate_max_pain():
    strikes = df_ce['strikePrice'].tolist()
    max_pain_strike = None
    min_total_oi = float('inf')
    
    for strike in strikes:
        ce_oi = df_ce[df_ce['strikePrice'] >= strike]['OI_CE'].sum()
        pe_oi = df_pe[df_pe['strikePrice'] <= strike]['OI_PE'].sum()
        total_oi = ce_oi + pe_oi
        if total_oi < min_total_oi:
            min_total_oi = total_oi
            max_pain_strike = strike
    return max_pain_strike

max_pain = calculate_max_pain()

# --- Auto-alert for extreme PCR ---
if "extreme_alerted" not in st.session_state:
    st.session_state.extreme_alerted = []

current_time = datetime.datetime.now().strftime('%H:%M')
if fresh_pcr > 1.5 and f"FRESH_HIGH-{current_time[:4]}" not in st.session_state.extreme_alerted:
    msg = f"🚨 EXTREME FRESH PCR HIGH: {fresh_pcr}\nSpot: {underlying}\nTime: {current_time}"
    if send_telegram_message(msg):
        st.session_state.extreme_alerted.append(f"FRESH_HIGH-{current_time[:4]}")

if fresh_pcr < 0.5 and f"FRESH_LOW-{current_time[:4]}" not in st.session_state.extreme_alerted:
    msg = f"🚨 EXTREME FRESH PCR LOW: {fresh_pcr}\nSpot: {underlying}\nTime: {current_time}"
    if send_telegram_message(msg):
        st.session_state.extreme_alerted.append(f"FRESH_LOW-{current_time[:4]}")

# --- Log to Supabase (with fresh data) ---
log_pcr(selected_expiry, underlying, atm_strike, pcr, total_ce_oi, total_pe_oi, fresh_pcr, fresh_ce_oi, fresh_pe_oi)

# --- Enhanced Dashboard ---
st.title("📊 NIFTY Option Screener – Fresh Intraday PCR")

# Primary metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("📈 NIFTY Spot", f"{underlying:.1f}")
with col2: st.metric("📉 Total PCR", pcr, delta=round(pcr_change,2))
with col3: st.metric("🔥 Fresh PCR", fresh_pcr, delta=round(fresh_pcr_change,2))
with col4: st.metric("📊 Volume PCR", pcr_vol)
with col5: st.metric("🎯 ATM Strike", atm_strike)

# Secondary metrics
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("💥 Max Pain", max_pain)
with col2: st.metric(f"📅 {current_session} Fresh PCR", session_fresh_pcr)
with col3: 
    fresh_sentiment = "🟢 Fresh Bullish" if fresh_pcr > 1.2 else "🔴 Fresh Bearish" if fresh_pcr < 0.8 else "🟡 Fresh Neutral"
    st.write(f"**{fresh_sentiment}**")
with col4: 
    if fresh_ce_oi > 0:
        st.metric("🆕 Fresh CE OI", f"{fresh_ce_oi:,}")
    if fresh_pe_oi > 0:
        st.metric("🆕 Fresh PE OI", f"{fresh_pe_oi:,}")

# Enhanced sentiment display
col1, col2 = st.columns(2)
with col1:
    if pcr > 1.2: st.success("🟢 Total PCR: Bullish")
    elif pcr < 0.8: st.error("🔴 Total PCR: Bearish")
    else: st.warning("🟡 Total PCR: Neutral")

with col2:
    if st.button("📨 Send Telegram Signal"):
        msg = f"""📊 <b>NIFTY PCR Alert</b>

💹 Spot: {underlying:.1f}
📉 Total PCR: {pcr}
🔥 Fresh PCR: {fresh_pcr}
📊 Volume PCR: {pcr_vol}
🎯 ATM: {atm_strike}
💥 Max Pain: {max_pain}
📅 {current_session} Session Fresh PCR: {session_fresh_pcr}
⏰ {datetime.datetime.now().strftime('%H:%M')}"""
        send_telegram_message(msg)

# --- Fresh OI Analysis Table ---
st.markdown("### 🔥 Fresh OI Analysis")
fresh_analysis = pd.DataFrame({
    "Metric": ["Fresh CE OI", "Fresh PE OI", "Fresh PCR", "Session Fresh PCR"],
    "Value": [f"{fresh_ce_oi:,}", f"{fresh_pe_oi:,}", fresh_pcr, session_fresh_pcr],
    "Interpretation": [
        "🔴 Bearish if high" if fresh_ce_oi > fresh_pe_oi else "🟢 Bullish signal",
        "🟢 Bullish if high" if fresh_pe_oi > fresh_ce_oi else "🔴 Bearish signal", 
        "🟢 Bullish" if fresh_pcr > 1.2 else "🔴 Bearish" if fresh_pcr < 0.8 else "🟡 Neutral",
        f"🟢 {current_session} Bullish" if session_fresh_pcr > 1.2 else f"🔴 {current_session} Bearish" if session_fresh_pcr < 0.8 else f"🟡 {current_session} Neutral"
    ]
})
st.dataframe(fresh_analysis, use_container_width=True)

# --- Merge & Display Option Chain ---
merged_df = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice")
merged_df["IV_CE"], merged_df["IV_PE"] = 12.5, 13.2
def price_tag(iv): return "🟢 Cheap" if iv<10 else "🔴 Expensive" if iv>15 else "🟡 Fair"
merged_df["Price_Tag_CE"] = merged_df["IV_CE"].apply(price_tag)
merged_df["Price_Tag_PE"] = merged_df["IV_PE"].apply(price_tag)

# Highlight ATM row and fresh OI changes
def highlight_rows(row):
    styles = [''] * len(row)
    if row["strikePrice"] == atm_strike:
        styles = ['background-color: yellow'] * len(row)
    
    # Highlight significant fresh OI
    if abs(row.get("Chg_OI_CE", 0)) > 10000:
        styles[merged_df.columns.get_loc("Chg_OI_CE")] = 'background-color: lightcoral'
    if abs(row.get("Chg_OI_PE", 0)) > 10000:
        styles[merged_df.columns.get_loc("Chg_OI_PE")] = 'background-color: lightgreen'
    
    return styles

st.markdown("### 🧾 Option Chain (Fresh OI Highlighted)")
st.markdown("*🟡 = ATM Strike, 🔴 = High CE OI Change, 🟢 = High PE OI Change*")
styled_df = merged_df.style.apply(highlight_rows, axis=1)
st.dataframe(styled_df, use_container_width=True)

# --- Enhanced Intraday Analysis with Fresh PCR ---
def fetch_today_logs():
    today = datetime.datetime.now().date().isoformat()
    res = supabase.table("nifty_pcr_logs").select("*").gte("timestamp", today).order("timestamp").execute()
    return pd.DataFrame(res.data)

logs_df = fetch_today_logs()
if not logs_df.empty:
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    
    def get_session(time):
        if datetime.time(9,15) <= time < datetime.time(12,30): return "Morning"
        elif datetime.time(12,30) <= time < datetime.time(14,30): return "Afternoon"
        elif datetime.time(14,30) <= time <= datetime.time(15,40): return "Closing"
        else: return "Off Hours"
    
    logs_df["session"] = logs_df["timestamp"].dt.time.apply(get_session)
    logs_df = logs_df[logs_df["session"] != "Off Hours"]

    st.markdown("### 📈 Intraday PCR Trends")
    
    # Dual chart - Total PCR vs Fresh PCR
    fig = go.Figure()
    
    # Total PCR
    fig.add_trace(go.Scatter(
        x=logs_df["timestamp"], 
        y=logs_df["pcr"], 
        mode='lines+markers', 
        name='Total PCR',
        line=dict(color='blue', width=2)
    ))
    
    # Fresh PCR (if available)
    if 'fresh_pcr' in logs_df.columns:
        fig.add_trace(go.Scatter(
            x=logs_df["timestamp"], 
            y=logs_df["fresh_pcr"], 
            mode='lines+markers', 
            name='Fresh PCR',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Reference lines
    fig.add_hline(y=1.2, line_dash="dash", line_color="green", annotation_text="Bullish")
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Bearish")
    
    fig.update_layout(
        height=500, 
        title="Total PCR vs Fresh PCR",
        xaxis_title="Time", 
        yaxis_title="PCR",
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Session Analysis
    if 'fresh_pcr' in logs_df.columns:
        session_stats = logs_df.groupby("session").agg({
            "pcr": "mean",
            "fresh_pcr": "mean"
        }).round(3)
        
        st.markdown("### 📊 Session-wise Analysis")
        col1, col2, col3 = st.columns(3)
        
        sessions = ["Morning", "Afternoon", "Closing"]
        cols = [col1, col2, col3]
        
        for i, session in enumerate(sessions):
            if session in session_stats.index:
                with cols[i]:
                    st.metric(f"🕐 {session} Total PCR", session_stats.loc[session, "pcr"])
                    st.metric(f"🔥 {session} Fresh PCR", session_stats.loc[session, "fresh_pcr"])
        
        # Enhanced Sentiment Flip Detection (using Fresh PCR)
        st.markdown("### 🔄 Fresh PCR Sentiment Flips")
        
        flips = []
        sessions_order = ["Morning", "Afternoon", "Closing"]
        
        if "fresh_sentiment_flips_alerted" not in st.session_state:
            st.session_state.fresh_sentiment_flips_alerted = []
        
        for i in range(len(sessions_order) - 1):
            s1, s2 = sessions_order[i], sessions_order[i + 1]
            
            if s1 in session_stats.index and s2 in session_stats.index:
                p1, p2 = session_stats.loc[s1, "fresh_pcr"], session_stats.loc[s2, "fresh_pcr"]
                flip_text = None
                
                if p1 > 1.1 and p2 < 0.9:
                    flip_text = f"⚠️ Fresh Bullish ({s1}) → Fresh Bearish ({s2})"
                elif p1 < 0.9 and p2 > 1.1:
                    flip_text = f"✅ Fresh Bearish ({s1}) → Fresh Bullish ({s2})"
                elif abs(p1 - p2) > 0.4:
                    direction = "📈" if p2 > p1 else "📉"
                    flip_text = f"{direction} Fresh PCR shift: {s1} ({p1:.2f}) → {s2} ({p2:.2f})"
                
                if flip_text:
                    flips.append(flip_text)
                    
                    # Auto-alert for major fresh flips
                    if flip_text not in st.session_state.fresh_sentiment_flips_alerted and ("Fresh Bullish → Fresh Bearish" in flip_text or "Fresh Bearish → Fresh Bullish" in flip_text):
                        message = f"""🚨 <b>Fresh PCR Flip Alert</b>
                        
{flip_text}
💹 Current Spot: {underlying:.2f}
⏰ Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
                        
                        if send_telegram_message(message):
                            st.session_state.fresh_sentiment_flips_alerted.append(flip_text)
                            st.success(f"📱 Fresh PCR Auto-alert sent: {flip_text}")
        
        for flip in flips:
            if "Fresh Bullish → Fresh Bearish" in flip or "Fresh Bearish → Fresh Bullish" in flip:
                st.warning(flip)
            else:
                st.info(flip)
        
        if not flips:
            st.success("✅ No significant fresh PCR flips detected today.")

else:
    st.info("📊 No intraday data available yet. Fresh PCR data will appear as the day progresses.")

# --- Enhanced Sidebar Info ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔥 Fresh PCR Status")
st.sidebar.write(f"⏰ Last Update: {datetime.datetime.now().strftime('%H:%M:%S')}")
st.sidebar.write(f"📈 Underlying: {underlying:.1f}")
st.sidebar.write(f"🎯 ATM Strike: {atm_strike}")
st.sidebar.write(f"💥 Max Pain: {max_pain}")
st.sidebar.write(f"📅 Current Session: {current_session}")
st.sidebar.write(f"🔥 Fresh PCR: {fresh_pcr}")
st.sidebar.write(f"📅 Session Fresh PCR: {session_fresh_pcr}")

st.sidebar.markdown("### 📊 PCR Levels")
st.sidebar.write("🟢 Bullish: PCR > 1.2")
st.sidebar.write("🔴 Bearish: PCR < 0.8")
st.sidebar.write("🟡 Neutral: 0.8-1.2")

st.sidebar.markdown("### 🔥 Fresh vs Total PCR")
st.sidebar.write("**Total PCR**: Overall market positioning")
st.sidebar.write("**Fresh PCR**: New positions today")
st.sidebar.write("**Session Fresh**: New positions this session")