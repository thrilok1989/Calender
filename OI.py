import streamlit as st
import pandas as pd
import requests
import time
import datetime
from supabase import create_client

# --- Streamlit Config ---
st.set_page_config(page_title="Nifty Option Screener with Intraday PCR", layout="wide")

# --- Auto refresh every 1 min ---
def auto_refresh(interval_sec=60):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh(60)

# --- Supabase Setup ---
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase = create_client(supabase_url, supabase_key)

# --- Log PCR data into Supabase ---
def log_pcr_to_supabase(expiry, underlying, atm_strike, pcr, total_ce_oi, total_pe_oi):
    try:
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "expiry": expiry,
            "underlying": underlying,
            "atm_strike": atm_strike,
            "pcr": pcr,
            "total_ce_oi": int(total_ce_oi),
            "total_pe_oi": int(total_pe_oi),
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
        st.warning("⚠️ Could not fetch data from NSE. Try again later.")
        return None

st.title("📊 NIFTY Option Screener – Intraday PCR Tracker")

# --- Fetch data ---
data = fetch_option_chain()
if data is None:
    st.stop()

raw_data = data['records']['data']
expiry_list = data['records']['expiryDates']
underlying = data['records'].get('underlyingValue', 0)

# --- Expiry selection ---
selected_expiry = st.selectbox("📅 Select Expiry Date", expiry_list)

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
            })
        if "PE" in item:
            pe = item["PE"]
            pe_list.append({
                "strikePrice": strike,
                "OI_PE": pe.get("openInterest", 0),
            })

df_ce = pd.DataFrame(ce_list)
df_pe = pd.DataFrame(pe_list)

# --- ATM Strike ---
atm_strike = min(df_ce['strikePrice'], key=lambda x: abs(x - underlying))

# --- Filter around ATM ---
df_ce = df_ce[(df_ce["strikePrice"] >= atm_strike - 500) & (df_ce["strikePrice"] <= atm_strike + 500)]
df_pe = df_pe[(df_pe["strikePrice"] >= atm_strike - 500) & (df_pe["strikePrice"] <= atm_strike + 500)]

# --- PCR Calculation ---
total_ce_oi = df_ce["OI_CE"].sum()
total_pe_oi = df_pe["OI_PE"].sum()
pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

st.metric("📉 Current PCR", pcr)

# --- Log into Supabase ---
log_pcr_to_supabase(selected_expiry, underlying, atm_strike, pcr, total_ce_oi, total_pe_oi)

# --- Fetch Today’s Logs from Supabase ---
def fetch_today_logs():
    today = datetime.datetime.now().date().isoformat()
    res = supabase.table("nifty_pcr_logs").select("*").gte("timestamp", today).execute()
    return pd.DataFrame(res.data)

logs_df = fetch_today_logs()

if not logs_df.empty:
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    logs_df["session"] = logs_df["timestamp"].apply(
        lambda x: "Morning" if x.time() < datetime.time(12, 30) else "Afternoon"
    )

    # --- PCR Trend Chart ---
    st.markdown("### 📈 Intraday PCR Trend")
    st.line_chart(logs_df.set_index("timestamp")["pcr"])

    # --- Morning vs Afternoon ---
    morning_pcr = logs_df[logs_df["session"]=="Morning"]["pcr"].mean()
    afternoon_pcr = logs_df[logs_df["session"]=="Afternoon"]["pcr"].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌅 Morning Avg PCR", round(morning_pcr,2) if not pd.isna(morning_pcr) else "-")
    with col2:
        st.metric("🌇 Afternoon Avg PCR", round(afternoon_pcr,2) if not pd.isna(afternoon_pcr) else "-")

    # --- Sentiment Shift Detection ---
    if morning_pcr and afternoon_pcr:
        if morning_pcr > 1 and afternoon_pcr < 0.8:
            st.warning("⚠️ Sentiment shifted: Bullish (Morning) → Bearish (Afternoon)")
        elif morning_pcr < 0.8 and afternoon_pcr > 1.2:
            st.success("✅ Sentiment shifted: Bearish (Morning) → Bullish (Afternoon)")
        else:
            st.info("ℹ️ No major sentiment shift detected yet.")
