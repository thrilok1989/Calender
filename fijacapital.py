import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import pytz  # for IST timezone

# -------------------------
# ✅ Market Hours Check
# -------------------------
def is_market_open():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    # NSE market open Mon-Fri 9:00 - 15:40
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=40, second=0, microsecond=0)
    return market_open <= now <= market_close

if not is_market_open():
    st.warning("⚠️ NSE Market is currently closed. This app runs only Monday–Friday, 9:00–15:40 IST.")
    st.stop()

st.set_page_config(page_title="Nifty Option Screener", layout="wide")

# 🔄 Auto refresh every 1 minutes
def auto_refresh(interval_sec=60):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh(60)

# Telegram Bot Setup
def send_telegram_message(message):
    try:
        bot_token = st.secrets["TELEGRAM"]["BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM"]["CHAT_ID"]
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to send Telegram message: {e}")
        return False

st.title("📊 NIFTY Option Screener – Fijacapital")
st.markdown("⏰ Auto-refresh every 1 minutes | 🔄 Live NSE Option Chain Analysis")

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

# 🔄 Fetch data
data = fetch_option_chain()
if data is None:
    st.stop()

raw_data = data['records']['data']
expiry_list = data['records']['expiryDates']
underlying = data['records'].get('underlyingValue', 0)

# 🗓️ Expiry Date Selection
selected_expiry = st.selectbox("📅 Select Expiry Date", expiry_list)

# Lists to hold CE and PE data
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

# 🎯 ATM Strike
atm_strike = min(df_ce['strikePrice'], key=lambda x: abs(x - underlying))

# Filter ±500 points around ATM
df_ce = df_ce[(df_ce["strikePrice"] >= atm_strike - 500) & (df_ce["strikePrice"] <= atm_strike + 500)]
df_pe = df_pe[(df_pe["strikePrice"] >= atm_strike - 500) & (df_pe["strikePrice"] <= atm_strike + 500)]

# 📉 PCR Calculation
total_ce_oi = df_ce["OI_CE"].sum()
total_pe_oi = df_pe["OI_PE"].sum()
pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

# Track PCR changes for signals
if "previous_pcr" not in st.session_state:
    st.session_state.previous_pcr = pcr

pcr_change = pcr - st.session_state.previous_pcr
st.session_state.previous_pcr = pcr

# 📊 PCR & TREND METER Display
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
    # Add a button to send trading signals
    if st.button("📨 Send Trading Signal via Telegram"):
        signal_message = f"📊 NIFTY Trading Signal\nPCR: {pcr}\nSpot: {underlying}\nATM: {atm_strike}\nTrend: {'Bullish' if pcr > 1.2 else 'Bearish' if pcr < 0.8 else 'Neutral'}"
        if send_telegram_message(signal_message):
            st.success("Signal sent successfully!")
        else:
            st.error("Failed to send signal. Check your Telegram credentials.")

# Merge CE and PE
merged_df = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice")

# 🧾 Display Table
st.markdown("### 🧾 Combined Option Chain (CALL + PUT Side-by-Side)")
st.caption(f"📍 Spot: `{underlying}` | 🎯 ATM Strike: `{atm_strike}` | 📅 Expiry: `{selected_expiry}`")

# 💸 Premium Analysis based on IV
def price_tag(iv):
    if iv < 10:
        return "🟢 Cheap"
    elif iv > 15:
        return "🔴 Expensive"
    else:
        return "🟡 Fair"

# Add dummy IV columns (can replace with live IV later)
merged_df["IV_CE"] = 12.5  # ← Placeholder IV (Updateable)
merged_df["IV_PE"] = 13.2

merged_df["Price_Tag_CE"] = merged_df["IV_CE"].apply(price_tag)
merged_df["Price_Tag_PE"] = merged_df["IV_PE"].apply(price_tag)

# 🎨 Highlight Cheap/Expensive tags with color
def highlight_tags(val):
    if val == '🟢 Cheap':
        return 'color: green'
    elif val == '🔴 Expensive':
        return 'color: red'
    elif val == '🟡 Fair':
        return 'color: orange'
    return ''

styled_df = merged_df.style.applymap(highlight_tags, subset=["Price_Tag_CE", "Price_Tag_PE"])

st.dataframe(styled_df, use_container_width=True)

# 🔍 Breakout Zones
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

# 🛑📈 Support & Resistance
df_ce_score = df_ce.copy()
df_pe_score = df_pe.copy()
df_ce_score["score"] = df_ce_score["Chg_OI_CE"] + df_ce_score["Vol_CE"]
df_pe_score["score"] = df_pe_score["Chg_OI_PE"] + df_pe_score["Vol_PE"]

resistance_strike = df_ce_score.sort_values("score", ascending=False).iloc[0]["strikePrice"]
support_strike = df_pe_score.sort_values("score", ascending=False).iloc[0]["strikePrice"]

st.markdown("### 🛑📈 Support & Resistance Zone")
col1, col2 = st.columns(2)
with col1:
    st.error(f"📉 Strong Support at **{int(support_strike)}** (PUT OI + Vol)")
with col2:
    st.success(f"📈 Strong Resistance at **{int(resistance_strike)}** (CALL OI + Vol)")

# 📊 Smart Sentiment Detector
st.markdown("### 🧠 Smart Sentiment Detector (Based on Premium + OI + IV)")

sentiments = []

# 1. Call Premium > Fair (assume Fair IV = 12.5)
if merged_df["IV_CE"].mean() > 15:
    sentiments.append("📈 **Call Premium > Fair Value** → বাজার Up যেতে পারে")

# 2. Put Premium > Fair
if merged_df["IV_PE"].mean() > 15:
    sentiments.append("📉 **Put Premium > Fair Value** → বাজার Down যেতে পারে")

# 3. IV rising
if merged_df["IV_CE"].mean() > 13 and merged_df["IV_PE"].mean() > 13:
    sentiments.append("⚠️ **IV বাড়ছে** → বড় মুভ আসতে পারে")

# 4. Call OI↑ + Premium↑ = Bullish
if df_ce["Chg_OI_CE"].sum() > 0 and df_ce["Vol_CE"].sum() > 0:
    sentiments.append("🟩 **Call OI↑ + Premium↑** → Smart money buying → Bullish")

# 5. Put OI↑ + Premium↑ = Bearish
if df_pe["Chg_OI_PE"].sum() > 0 and df_pe["Vol_PE"].sum() > 0:
    sentiments.append("🟥 **Put OI↑ + Premium↑** → Smart money buying → Bearish")

# 6. Call premium↑, OI↓ = Short covering
if df_ce["Vol_CE"].sum() > 0 and df_ce["Chg_OI_CE"].sum() < 0:
    sentiments.append("🔼 **Call প্রিমিয়াম↑, OI↓** → Short covering → Up move")

# 7. Put premium↑, OI↓ = Put covering → bounce
if df_pe["Vol_PE"].sum() > 0 and df_pe["Chg_OI_PE"].sum() < 0:
    sentiments.append("🔼 **Put প্রিমিয়াম↑, OI↓** → Short covering → Up bounce")

# 8. PCR Signal
if pcr > 1.3:
    sentiments.append("📗 **PCR > 1.3** → Bullish bias")
elif pcr < 0.7:
    sentiments.append("📕 **PCR < 0.7** → Bearish bias")

# Final Output
if sentiments:
    for s in sentiments:
        st.info(s)
else:
    st.warning("🤔 পর্যাপ্ত তথ্য নেই বাজার বিশ্লেষণের জন্য।")

# 🤖 Auto Trade Suggestion
st.markdown("### 🤖 Auto Trade Suggestion")

if support_strike == resistance_strike:
    action = "⚠️ Avoid Trade"
    reason = f"Support & Resistance both at {support_strike} → Range-bound zone"
elif resistance_strike > support_strike and pcr < 0.8:
    action = "🟥 BUY PUT"
    reason = f"PCR = {pcr} (Bearish) & Resistance > Support → Downside expected"
elif support_strike > resistance_strike and pcr > 1.2:
    action = "🟩 BUY CALL"
    reason = f"PCR = {pcr} (Bullish) & Support > Resistance → Upside expected"
else:
    action = "🔄 Wait or Sell Options"
    reason = f"Indecisive structure – PCR = {pcr} | Support ≠ Resistance"

st.info(f"**Suggested Action:** {action}")
st.caption(f"📌 Reason: {reason}")

# Auto-send signal when strong signal detected
if (pcr > 1.3 or pcr < 0.7) and ("last_signal_sent" not in st.session_state or 
                                 time.time() - st.session_state.last_signal_sent > 3600):  # 1 hour cooldown
    signal_message = f"🚨 STRONG NIFTY SIGNAL\nPCR: {pcr} ({'Bullish' if pcr > 1.3 else 'Bearish'})\nSpot: {underlying}\nSuggested: {action}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    if send_telegram_message(signal_message):
        st.session_state.last_signal_sent = time.time()
        st.success("Strong signal automatically sent to Telegram!")
