import streamlit as st
import pandas as pd
import requests
import time
import datetime
from supabase import create_client

# --- Streamlit Config ---
st.set_page_config(page_title="Nifty Option Screener – Intraday PCR", layout="wide")

# --- Auto refresh every 1 min ---
def auto_refresh(interval_sec=60):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh(60)

# --- Telegram Bot Setup ---
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

# --- Function to log PCR to Supabase ---
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

# --- Fetch Data ---
data = fetch_option_chain()
if data is None:
    st.stop()

raw_data = data['records']['data']
expiry_list = data['records']['expiryDates']
underlying = data['records'].get('underlyingValue', 0)

# --- Expiry Selection ---
selected_expiry = st.selectbox("📅 Select Expiry Date", expiry_list)

# --- CE & PE Lists ---
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

# --- ATM Strike ---
atm_strike = min(df_ce['strikePrice'], key=lambda x: abs(x - underlying))

# --- Filter ±500 points around ATM ---
df_ce = df_ce[(df_ce["strikePrice"] >= atm_strike - 500) & (df_ce["strikePrice"] <= atm_strike + 500)]
df_pe = df_pe[(df_pe["strikePrice"] >= atm_strike - 500) & (df_pe["strikePrice"] <= atm_strike + 500)]

# --- PCR Calculation ---
total_ce_oi = df_ce["OI_CE"].sum()
total_pe_oi = df_pe["OI_PE"].sum()
pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

# --- Track PCR Changes ---
if "previous_pcr" not in st.session_state:
    st.session_state.previous_pcr = pcr
pcr_change = pcr - st.session_state.previous_pcr
st.session_state.previous_pcr = pcr

# --- Log PCR to Supabase ---
log_pcr_to_supabase(selected_expiry, underlying, atm_strike, pcr, total_ce_oi, total_pe_oi)

# --- Dashboard: PCR Metric & Trend ---
st.title("📊 NIFTY Option Screener – Fijacapital")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📉 Put Call Ratio (PCR)", value=pcr, delta=round(pcr_change, 2))
with col2:
    if pcr > 1.2:
        st.success("🟢 Bullish Sentiment")
    elif pcr < 0.8:
        st.error("🔴 Bearish Sentiment")
    else:
        st.warning("🟡 Neutral Sentiment")
with col3:
    if st.button("📨 Send Trading Signal via Telegram"):
        signal_message = f"📊 NIFTY Trading Signal\nPCR: {pcr}\nSpot: {underlying}\nATM: {atm_strike}\nTrend: {'Bullish' if pcr > 1.2 else 'Bearish' if pcr < 0.8 else 'Neutral'}"
        if send_telegram_message(signal_message):
            st.success("Signal sent successfully!")
        else:
            st.error("Failed to send signal. Check your Telegram credentials.")

# --- Merge CE & PE for Table ---
merged_df = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice")

# --- Combined Option Chain Table ---
st.markdown("### 🧾 Combined Option Chain (CALL + PUT)")
st.caption(f"📍 Spot: `{underlying}` | 🎯 ATM Strike: `{atm_strike}` | 📅 Expiry: `{selected_expiry}`")

def price_tag(iv):
    if iv < 10: return "🟢 Cheap"
    elif iv > 15: return "🔴 Expensive"
    else: return "🟡 Fair"

merged_df["IV_CE"], merged_df["IV_PE"] = 12.5, 13.2
merged_df["Price_Tag_CE"] = merged_df["IV_CE"].apply(price_tag)
merged_df["Price_Tag_PE"] = merged_df["IV_PE"].apply(price_tag)

def highlight_tags(val):
    if val == '🟢 Cheap': return 'color: green'
    elif val == '🔴 Expensive': return 'color: red'
    elif val == '🟡 Fair': return 'color: orange'
    return ''

styled_df = merged_df.style.applymap(highlight_tags, subset=["Price_Tag_CE", "Price_Tag_PE"])
st.dataframe(styled_df, use_container_width=True)

# --- Breakout Zones ---
df_ce_top = df_ce.sort_values(by="Chg_OI_CE", ascending=False).head(3)
df_pe_top = df_pe.sort_values(by="Chg_OI_PE", ascending=False).head(3)
st.markdown("### 🔍 Breakout Zones")
col1, col2 = st.columns(2)
with col1:
    st.subheader("🚀 Top CALL Breakout")
    for _, row in df_ce_top.iterrows():
        st.success(f"Strike: {row['strikePrice']} | OI↑: {int(row['Chg_OI_CE'])} | Vol: {int(row['Vol_CE'])}")
with col2:
    st.subheader("🔻 Top PUT Breakout")
    for _, row in df_pe_top.iterrows():
        st.success(f"Strike: {row['strikePrice']} | OI↑: {int(row['Chg_OI_PE'])} | Vol: {int(row['Vol_PE'])}")

# --- Support & Resistance ---
df_ce_score, df_pe_score = df_ce.copy(), df_pe.copy()
df_ce_score["score"] = df_ce_score["Chg_OI_CE"] + df_ce_score["Vol_CE"]
df_pe_score["score"] = df_pe_score["Chg_OI_PE"] + df_pe_score["Vol_PE"]

resistance_strike = df_ce_score.sort_values("score", ascending=False).iloc[0]["strikePrice"]
support_strike = df_pe_score.sort_values("score", ascending=False).iloc[0]["strikePrice"]

st.markdown("### 🛑📈 Support & Resistance Zone")
col1, col2 = st.columns(2)
with col1: st.error(f"📉 Strong Support at **{int(support_strike)}**")
with col2: st.success(f"📈 Strong Resistance at **{int(resistance_strike)}**")

# --- Intraday PCR Trend from Supabase ---
def fetch_today_logs():
    today = datetime.datetime.now().date().isoformat()
    res = supabase.table("nifty_pcr_logs").select("*").gte("timestamp", today).execute()
    return pd.DataFrame(res.data)

logs_df = fetch_today_logs()
if not logs_df.empty:
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    logs_df["session"] = logs_df["timestamp"].apply(lambda x: "Morning" if x.time() < datetime.time(12,30) else "Afternoon")
    st.markdown("### 📈 Intraday PCR Trend")
    st.line_chart(logs_df.set_index("timestamp")["pcr"])

    morning_pcr = logs_df[logs_df["session"]=="Morning"]["pcr"].mean()
    afternoon_pcr = logs_df[logs_df["session"]=="Afternoon"]["pcr"].mean()
    col1, col2 = st.columns(2)
    with col1: st.metric("🌅 Morning Avg PCR", round(morning_pcr,2) if not pd.isna(morning_pcr) else "-")
    with col2: st.metric("🌇 Afternoon Avg PCR", round(afternoon_pcr,2) if not pd.isna(afternoon_pcr) else "-")

    # Detect Sentiment Shift
    if morning_pcr and afternoon_pcr:
        if morning_pcr > 1 and afternoon_pcr < 0.8:
            st.warning("⚠️ Sentiment shifted: Bullish (Morning) → Bearish (Afternoon)")
        elif morning_pcr < 0.8 and afternoon_pcr > 1.2:
            st.success("✅ Sentiment shifted: Bearish (Morning) → Bullish (Afternoon)")
        else:
            st.info("ℹ️ No major sentiment shift detected yet.")
