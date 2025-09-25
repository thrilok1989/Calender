import streamlit as st
import pandas as pd
import requests
import time
import datetime
from supabase import create_client

# --- Streamlit Config ---
st.set_page_config(page_title="Nifty Option Screener – Intraday PCR", layout="wide")

# --- Market Hours Check ---
def is_market_open():
    now_utc = datetime.datetime.utcnow()
    ist_offset = datetime.timedelta(hours=5, minutes=30)
    now_ist = now_utc + ist_offset
    # Weekday: Monday=0, Friday=4
    if now_ist.weekday() > 4:
        return False
    market_open = datetime.time(9, 0)
    market_close = datetime.time(15, 40)
    return market_open <= now_ist.time() <= market_close

if not is_market_open():
    st.warning("⛔ Market is closed. Data fetch only runs Monday-Friday 9:00–15:40 IST.")
    st.stop()

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

# --- Log PCR to Supabase ---
def log_pcr(expiry, underlying, atm_strike, pcr, total_ce_oi, total_pe_oi):
    try:
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "expiry": expiry,
            "underlying": underlying,
            "atm_strike": atm_strike,
            "pcr": pcr,
            "total_ce_oi": int(total_ce_oi),
            "total_pe_oi": int(total_pe_oi)
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

# --- ATM Strike & Filter ±500 ---
atm_strike = min(df_ce['strikePrice'], key=lambda x: abs(x - underlying))
df_ce = df_ce[(df_ce["strikePrice"] >= atm_strike - 500) & (df_ce["strikePrice"] <= atm_strike + 500)]
df_pe = df_pe[(df_pe["strikePrice"] >= atm_strike - 500) & (df_pe["strikePrice"] <= atm_strike + 500)]

# --- PCR Calculation ---
total_ce_oi = df_ce["OI_CE"].sum()
total_pe_oi = df_pe["OI_PE"].sum()
pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
if "previous_pcr" not in st.session_state: st.session_state.previous_pcr = pcr
pcr_change = pcr - st.session_state.previous_pcr
st.session_state.previous_pcr = pcr

# --- Log to Supabase ---
log_pcr(selected_expiry, underlying, atm_strike, pcr, total_ce_oi, total_pe_oi)

# --- Dashboard PCR & Trend ---
st.title("📊 NIFTY Option Screener – Intraday PCR")
col1, col2, col3 = st.columns(3)
with col1: st.metric(label="📉 PCR", value=pcr, delta=round(pcr_change,2))
with col2:
    if pcr > 1.2: st.success("🟢 Bullish")
    elif pcr < 0.8: st.error("🔴 Bearish")
    else: st.warning("🟡 Neutral")
with col3:
    if st.button("📨 Send Telegram Signal"):
        msg = f"📊 NIFTY PCR Signal\nPCR:{pcr}\nSpot:{underlying}\nATM:{atm_strike}"
        send_telegram_message(msg)

# --- Merge & Display Option Chain ---
merged_df = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice")
merged_df["IV_CE"], merged_df["IV_PE"] = 12.5, 13.2
def price_tag(iv): return "🟢 Cheap" if iv<10 else "🔴 Expensive" if iv>15 else "🟡 Fair"
merged_df["Price_Tag_CE"] = merged_df["IV_CE"].apply(price_tag)
merged_df["Price_Tag_PE"] = merged_df["IV_PE"].apply(price_tag)
st.markdown("### 🧾 Combined Option Chain")
st.dataframe(merged_df, use_container_width=True)

# --- Intraday PCR Trend (3 Sessions) ---
def fetch_today_logs():
    today = datetime.datetime.now().date().isoformat()
    res = supabase.table("nifty_pcr_logs").select("*").gte("timestamp", today).execute()
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

    st.markdown("### 📈 Intraday PCR Trend (3 Sessions)")
    st.line_chart(logs_df.set_index("timestamp")["pcr"])

    session_avg = logs_df.groupby("session")["pcr"].mean()
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🌅 Morning Avg PCR", round(session_avg.get("Morning",0),2))
    with col2: st.metric("🌇 Afternoon Avg PCR", round(session_avg.get("Afternoon",0),2))
    with col3: st.metric("🌙 Closing Avg PCR", round(session_avg.get("Closing",0),2))

    # --- Sentiment Flip Detection + Auto Telegram ---
    flips = []
    sessions_order = ["Morning","Afternoon","Closing"]
    if "sentiment_flips_alerted" not in st.session_state:
        st.session_state.sentiment_flips_alerted = []

    for i in range(len(sessions_order)-1):
        s1, s2 = sessions_order[i], sessions_order[i+1]
        p1, p2 = session_avg.get(s1,0), session_avg.get(s2,0)
        flip_text = None

        if p1 > 1 and p2 < 0.8:
            flip_text = f"⚠️ Bullish ({s1}) → Bearish ({s2})"
        elif p1 < 0.8 and p2 > 1.2:
            flip_text = f"✅ Bearish ({s1}) → Bullish ({s2})"

        if flip_text:
            flips.append(flip_text)
            if flip_text not in st.session_state.sentiment_flips_alerted:
                message = f"🚨 NIFTY Intraday PCR Flip Detected\n{flip_text}\nSpot: {underlying}\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                if send_telegram_message(message):
                    st.session_state.sentiment_flips_alerted.append(flip_text)
                    st.success(f"Telegram alert sent: {flip_text}")
                else:
                    st.error(f"Failed to send Telegram alert: {flip_text}")

    for f in flips: st.warning(f)
    if not flips: st.info("ℹ️ No major sentiment flips detected today.")
